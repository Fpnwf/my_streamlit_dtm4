{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0de331ef-7b91-4332-b6b5-db82ec11019c",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'streamlit'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mModuleNotFoundError\u001b[39m                       Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 1\u001b[39m\n\u001b[32m----> \u001b[39m\u001b[32m1\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mstreamlit\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mst\u001b[39;00m\n\u001b[32m      2\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mpandas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mpd\u001b[39;00m\n\u001b[32m      3\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mmatplotlib\u001b[39;00m\u001b[34;01m.\u001b[39;00m\u001b[34;01mpyplot\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mplt\u001b[39;00m\n",
      "\u001b[31mModuleNotFoundError\u001b[39m: No module named 'streamlit'"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "# ตั้งค่าหน้าเว็บ\n",
    "st.set_page_config(page_title=\"Flight Route Clustering\", layout=\"wide\")\n",
    "\n",
    "# โหลดข้อมูล (เปลี่ยน path ถ้าจำเป็น)\n",
    "df = pd.read_csv(\"Air_Traffic_Passenger_Statistics.csv\")\n",
    "\n",
    "# แปลงวันที่\n",
    "df['Activity Period'] = pd.to_datetime(df['Activity Period'], errors='coerce', format='%Y%m')\n",
    "df = df.dropna(subset=[\"Activity Period\"])\n",
    "\n",
    "# สร้าง Year, Month, Route\n",
    "df['Year'] = df['Activity Period'].dt.year\n",
    "df['Month'] = df['Activity Period'].dt.month\n",
    "df['Route'] = df['Operating Airline'] + \" to \" + df['GEO Region']\n",
    "\n",
    "# ดึงข้อมูลเฉพาะที่ต้องใช้\n",
    "st.title(\"✈️ Flight Route Clustering for Optimization\")\n",
    "\n",
    "# Sidebar\n",
    "with st.sidebar:\n",
    "    st.header(\"🔧 Clustering Configuration\")\n",
    "    k = st.slider(\"Select number of clusters (k)\", 2, 10, 3)\n",
    "\n",
    "# รวมข้อมูลต่อ route\n",
    "route_group = df.groupby(\"Route\").agg(\n",
    "    avg_passenger_per_month=(\"Passenger Count\", \"mean\"),\n",
    "    std_passenger=(\"Passenger Count\", \"std\"),\n",
    "    total_passenger=(\"Passenger Count\", \"sum\")\n",
    ").reset_index()\n",
    "\n",
    "# ลบค่าขาดหาย\n",
    "route_group.dropna(inplace=True)\n",
    "\n",
    "# ทำ Standardize ข้อมูลก่อน Clustering\n",
    "features = ['avg_passenger_per_month', 'std_passenger', 'total_passenger']\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(route_group[features])\n",
    "\n",
    "# ทำ KMeans\n",
    "kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
    "route_group['Cluster'] = kmeans.fit_predict(X_scaled)\n",
    "\n",
    "# PCA for visualization\n",
    "pca = PCA(n_components=2)\n",
    "pca_result = pca.fit_transform(X_scaled)\n",
    "route_group['PCA1'] = pca_result[:, 0]\n",
    "route_group['PCA2'] = pca_result[:, 1]\n",
    "\n",
    "# แสดงผล\n",
    "st.subheader(\"📊 Clustered Flight Routes\")\n",
    "st.dataframe(route_group[['Route', 'avg_passenger_per_month', 'std_passenger', 'total_passenger', 'Cluster']])\n",
    "\n",
    "# กราฟ Scatter Plot\n",
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "sns.scatterplot(data=route_group, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=100)\n",
    "plt.title(\"Cluster Visualization (PCA)\")\n",
    "plt.xlabel(\"PCA 1\")\n",
    "plt.ylabel(\"PCA 2\")\n",
    "st.pyplot(fig)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
