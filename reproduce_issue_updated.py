import pytest
import boto3
from moto import mock_secretsmanager
from moto.core import DEFAULT_ACCOUNT_ID


def setup_secretsmanager():
    with mock_secretsmanager():
        client = boto3.client("secretsmanager", region_name="us-east-1")
        name = "test-secret"
        client.create_secret(Name=name, SecretString="42")
        return client, name


def test_partial_arn():
    client, name = setup_secretsmanager()
    partial_arn = f"arn:aws:secretsmanager:us-east-1:{DEFAULT_ACCOUNT_ID}:secret:{name}"
    try:
        secret_value = client.get_secret_value(SecretId=partial_arn)
        print("Successfully retrieved secret.", secret_value)
    except Exception as e:
        print("Error occurred:", e)


test_partial_arn()