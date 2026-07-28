from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

default_args = {
    "owner": "lakshmi",
    "retries": 1,
}

with DAG(
    dag_id="ai_news_pipeline",
    default_args=default_args,
    description="Daily pipeline: fetch news -> clean -> embed & store",
    schedule="@daily",
    start_date=datetime(2026, 7, 28),
    catchup=False,
    tags=["rag", "news", "ai"],
) as dag:

    fetch_task = BashOperator(
        task_id="fetch_news",
        bash_command="cd /workspaces/DE-to-MLE-Journey && python ingestion/fetch_news.py",
    )

    clean_task = BashOperator(
        task_id="clean_articles",
        bash_command="cd /workspaces/DE-to-MLE-Journey && python transform/clean_articles.py",
    )

    embed_task = BashOperator(
        task_id="embed_and_store",
        bash_command="cd /workspaces/DE-to-MLE-Journey && python embed/embed_and_store.py",
    )

    fetch_task >> clean_task >> embed_task