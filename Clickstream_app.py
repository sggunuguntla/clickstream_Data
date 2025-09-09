import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except:
    SMOTE_AVAILABLE = False

# DATA GENERATION

def create_data(num_sessions=2000, max_clicks_per_session=8):
    rng = np.random.default_rng(42)
    rows = []
    session_id = 1000
    for _ in range(num_sessions):
        clicks = int(rng.integers(1, max_clicks_per_session+1))
        year = 2008
        month = int(rng.integers(4, 9))
        day = int(rng.integers(1, 29))
        country = int(rng.integers(1, 48))
        main_cat = int(rng.integers(1, 5))
        colour = int(rng.integers(1, 15))
        location = int(rng.integers(1, 7))
        model_photo = int(rng.integers(1, 3))
        price_flag = int(rng.choice([0,1]))
        page2_model = int(rng.integers(1, 218))

        for order in range(1, clicks+1):
            price = float(rng.uniform(5, 500))
            page = int(rng.integers(1, 6))
            rows.append({
                'YEAR': year,
                'MONTH': month,
                'DAY': day,
                'ORDER': order,
                'COUNTRY': country,
                'SESSION_ID': session_id,
                'PAGE_1_MAIN_CATEGORY': main_cat,
                'PAGE_2_CLOTHING_MODEL': page2_model,
                'COLOUR': colour,
                'LOCATION': location,
                'MODEL_PHOTOGRAPHY': model_photo,
                'PRICE': round(price, 2),
                'PRICE_2_HIGHER_THAN_AVG': price_flag,
                'PAGE': page,
            })
        session_id += 1

    df = pd.DataFrame(rows)

    # Inject missing values
    for col in ['COUNTRY','PRICE']:
        df.loc[df.sample(frac=0.015).index, col] = np.nan

    # Date features
    df['DATE'] = pd.to_datetime(df[['YEAR','MONTH','DAY']])
    df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek

    # Session features
    df['SESSION_LENGTH'] = df.groupby('SESSION_ID')['ORDER'].transform('max')
    df['BOUNCE'] = (df['SESSION_LENGTH']==1).astype(int)
    revisits = (df.groupby('SESSION_ID')['DAY'].transform('nunique') > 1).astype(int)
    df['REVISIT'] = revisits

    # Target labels
    df['PURCHASE_COMPLETED'] = np.where(
        (df['PRICE'].fillna(df['PRICE'].median())<120) &
        (df['SESSION_LENGTH']>=3) & (df['PAGE']<=3), 1, 0
    )

    # Revenue calculation
    df['REVENUE'] = (df['PRICE'].fillna(df['PRICE'].median()) * (1+0.05*(df['SESSION_LENGTH']-1))).round(2)

    return df

# STREAMLIT APP

st.set_page_config(page_title="Clickstream: Conversion & Revenue Predictor", layout="wide")
st.title("🛒 Clickstream Conversion Analysis")

df = create_data()

# Tabs
eda_tab, cls_tab, reg_tab, clust_tab, insight_tab, summary_tab, pred_tab = st.tabs(
    ["EDA","Classification","Regression","Clustering","Insights","Summary","Predictions"]
)

# EDA TAB
with eda_tab:
    st.subheader("Exploratory Data Analysis")
    st.write(df.head())

    fig, ax = plt.subplots(1,3, figsize=(12,3))
    for i,col in enumerate(['PRICE','ORDER','PAGE']):
        sns.histplot(df[col], bins=30, ax=ax[i])
    st.pyplot(fig)

    st.bar_chart(df['COUNTRY'].value_counts().head(10))

    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    rate = df.groupby('DAY_OF_WEEK')['PURCHASE_COMPLETED'].mean()
    st.line_chart(rate)

# COMMON PREPROCESSING

num_features = ['ORDER','PRICE','PAGE','DAY_OF_WEEK','SESSION_LENGTH']
cat_features = ['COUNTRY','PAGE_1_MAIN_CATEGORY','COLOUR','LOCATION','MODEL_PHOTOGRAPHY','PRICE_2_HIGHER_THAN_AVG']

pre = ColumnTransformer([
    ('num', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_features),

    ('cat', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features)
])

X = df[num_features+cat_features]
y_cls = df['PURCHASE_COMPLETED']
y_reg = df['REVENUE']

# CLASSIFICATION TAB

with cls_tab:
    st.subheader("Classification Models")
    X_train, X_val, y_train, y_val = train_test_split(X,y_cls,test_size=0.2, stratify=y_cls, random_state=42)

    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42)
        X_train_proc = pre.fit_transform(X_train)
        X_train_bal, y_train_bal = sm.fit_resample(X_train_proc, y_train)
    else:
        X_train_bal = pre.fit_transform(X_train)
        y_train_bal = y_train

    X_val_proc = pre.transform(X_val)

    cls_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)
    }
    if XGB_AVAILABLE:
        cls_models['XGBoost'] = XGBClassifier(eval_metric='logloss', use_label_encoder=False)

    cls_results = {}
    for name, model in cls_models.items():
        model.fit(X_train_bal, y_train_bal)
        preds = model.predict(X_val_proc)
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, pos_label=1)
        rec = recall_score(y_val, preds, pos_label=1)
        f1 = f1_score(y_val, preds, pos_label=1)
        auc = np.nan
        if hasattr(model,'predict_proba'):
            probs = model.predict_proba(X_val_proc)
            auc = roc_auc_score(y_val, probs[:,1])
        cls_results[name] = dict(acc=acc,prec=prec,rec=rec,f1=f1,auc=auc)

    st.dataframe(pd.DataFrame(cls_results).T)

# REGRESSION TAB

with reg_tab:
    st.subheader("Regression Models")
    X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X,y_reg,test_size=0.2, random_state=42)
    X_train_rp = pre.fit_transform(X_train_r)
    X_val_rp = pre.transform(X_val_r)

    reg_models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Gradient Boosting': GradientBoostingRegressor()
    }

    reg_results = {}
    for name, model in reg_models.items():
        model.fit(X_train_rp, y_train_r)
        preds = model.predict(X_val_rp)
        rmse = mean_squared_error(y_val_r, preds, squared=False)
        mae = mean_absolute_error(y_val_r, preds)
        r2 = r2_score(y_val_r, preds)
        reg_results[name] = dict(rmse=rmse,mae=mae,r2=r2)

    st.dataframe(pd.DataFrame(reg_results).T)

# CLUSTERING TAB

with clust_tab:
    st.subheader("Clustering")

    clust_features = ['ORDER','PRICE','PAGE_1_MAIN_CATEGORY','COLOUR']
    cl_pre = ColumnTransformer([
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())]), ['ORDER','PRICE']),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder())]), ['PAGE_1_MAIN_CATEGORY','COLOUR'])
    ])
    Xc = cl_pre.fit_transform(df[clust_features])

    km = KMeans(n_clusters=3, random_state=42)
    labels = km.fit_predict(Xc)

    # Metrics
    sil = silhouette_score(Xc, labels)
    db = davies_bouldin_score(Xc.toarray() if hasattr(Xc,'toarray') else Xc, labels)

    # Add cluster labels back
    df['Cluster'] = labels

    st.markdown(f"**Silhouette Score:** {sil:.3f}")
    st.markdown(f"**Davies–Bouldin Index:** {db:.3f}")

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(x=df['ORDER'], y=df['PRICE'], hue=df['Cluster'], palette='Set2', ax=ax)
    ax.set_title("Clusters by Order vs Price")
    st.pyplot(fig)

    # Show cluster profiles
    st.subheader("Cluster Profiles")
    cluster_summary = df.groupby('Cluster').agg(
        avg_order=('ORDER','mean'),
        avg_price=('PRICE','mean'),
        most_common_category=('PAGE_1_MAIN_CATEGORY', lambda x:x.mode()[0])
    ).round(2)
    st.dataframe(cluster_summary)

# INSIGHTS TAB

with insight_tab:
    st.subheader("Business Insights")

    best_model = LogisticRegression(max_iter=1000,class_weight='balanced').fit(pre.fit_transform(X), y_cls)
    preds_all = best_model.predict(pre.transform(X))
    df['PRED_PURCHASE'] = preds_all
    churn_like = df[df['PRED_PURCHASE']==0]

    st.write("At-risk events:",len(churn_like))

    labels_all = km.predict(Xc)
    df['Cluster'] = labels_all
    rec = df.groupby('Cluster').agg(
        avg_price=('PRICE','mean'),
        top_category=('PAGE_1_MAIN_CATEGORY', lambda s:s.value_counts().idxmax())
    )
    st.dataframe(rec)

# SUMMARY TAB

with summary_tab:
    st.subheader("Final Summary")

    cls_df = pd.DataFrame(cls_results).T
    reg_df = pd.DataFrame(reg_results).T

    best_cls = cls_df.sort_values(by='f1', ascending=False).iloc[0]
    best_reg = reg_df.sort_values(by='r2', ascending=False).iloc[0]

    summary_text = f"""
    **Classification:** Best model = **{cls_df.sort_values(by='f1', ascending=False).index[0]}**  
    Accuracy={best_cls['acc']:.2f}, F1={best_cls['f1']:.2f}, AUC={best_cls['auc']:.2f}

    **Regression:** Best model = **{reg_df.sort_values(by='r2', ascending=False).index[0]}**  
    R²={best_reg['r2']:.2f}, RMSE={best_reg['rmse']:.2f}

    **Clustering:** KMeans (3 clusters) → Silhouette={sil:.2f}, Davies–Bouldin={db:.2f}

    **Business Insights:** Identified {len(churn_like)} at-risk events. Recommendations available per cluster.
    """

    st.markdown(summary_text)

# PREDICTIONS TAB

with pred_tab:
    st.subheader("Make Predictions")

    order = st.number_input("Number of Orders (click depth)", min_value=1, max_value=100, value=3)
    price = st.number_input("Price of item", min_value=1.0, max_value=1000.0, value=50.0)
    page = st.number_input("Page Number", min_value=1, max_value=10, value=1)
    country = st.number_input("Country ID", min_value=1, max_value=47, value=5)
    main_cat = st.number_input("Main Category", min_value=1, max_value=5, value=2)
    colour = st.number_input("Colour ID", min_value=1, max_value=15, value=3)
    location = st.number_input("Location ID", min_value=1, max_value=7, value=2)
    model_photo = st.number_input("Model Photography", min_value=1, max_value=2, value=1)
    price_flag = st.selectbox("Is Price Higher than Avg?", [0,1])
    day_of_week = st.selectbox("Day of Week", list(range(7)))
    session_length = st.number_input("Session Length", min_value=1, max_value=10, value=3)

    input_df = pd.DataFrame([{
        'ORDER': order,
        'PRICE': price,
        'PAGE': page,
        'DAY_OF_WEEK': day_of_week,
        'SESSION_LENGTH': session_length,
        'COUNTRY': country,
        'PAGE_1_MAIN_CATEGORY': main_cat,
        'COLOUR': colour,
        'LOCATION': location,
        'MODEL_PHOTOGRAPHY': model_photo,
        'PRICE_2_HIGHER_THAN_AVG': price_flag
    }])

    if st.button("Predict"):
        X_proc = pre.transform(input_df)

        # Best Classifier
        cls_df = pd.DataFrame(cls_results).T
        best_cls_name = cls_df.sort_values(by='f1', ascending=False).index[0]
        if best_cls_name == "Logistic Regression":
            best_cls_model = LogisticRegression(max_iter=1000,class_weight='balanced')
        elif best_cls_name == "Random Forest":
            best_cls_model = RandomForestClassifier(n_estimators=200, random_state=42)
        elif best_cls_name == "XGBoost" and XGB_AVAILABLE:
            best_cls_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        else:
            best_cls_model = LogisticRegression(max_iter=1000,class_weight='balanced')
        best_cls_model.fit(pre.fit_transform(X), y_cls)
        purchase_pred = best_cls_model.predict(X_proc)[0]
        purchase_prob = best_cls_model.predict_proba(X_proc)[0][1]

        # Best Regressor
        reg_df = pd.DataFrame(reg_results).T
        best_reg_name = reg_df.sort_values(by='r2', ascending=False).index[0]
        if best_reg_name == "Linear":
            best_reg_model = LinearRegression()
        elif best_reg_name == "Ridge":
            best_reg_model = Ridge()
        elif best_reg_name == "Lasso":
            best_reg_model = Lasso()
        elif best_reg_name == "Gradient Boosting":
            best_reg_model = GradientBoostingRegressor()
        else:
            best_reg_model = GradientBoostingRegressor()
        best_reg_model.fit(pre.fit_transform(X), y_reg)
        revenue_pred = best_reg_model.predict(X_proc)[0]

        # Show results
        st.markdown(f"**Purchase Prediction:** {'Yes' if purchase_pred==1 else 'No'}")
        st.progress(int(purchase_prob*100))
        st.info(f"**Expected Revenue:** ${revenue_pred:.2f}")
        st.caption(f"Models used → Classifier: {best_cls_name}, Regressor: {best_reg_name}")

        # Download single prediction
        result_df = input_df.copy()
        result_df['Predicted_Purchase'] = "Yes" if purchase_pred==1 else "No"
        result_df['Purchase_Probability'] = round(purchase_prob,2)
        result_df['Expected_Revenue'] = round(revenue_pred,2)

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prediction as CSV",
            data=csv,
            file_name="clickstream_prediction.csv",
            mime="text/csv"
        )