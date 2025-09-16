from moto import mock_secretsmanager
import boto3
from moto.core import DEFAULT_ACCOUNT_ID


def test_partial_arn():
    with mock_secretsmanager():
        client = boto3.client("secretsmanager", region_name="us-east-1")
        # create test secret
        name = "test-secret"
        client.create_secret(Name=name, SecretString="42")

        # try to retrieve using partial ARN
        partial_arn = f"arn:aws:secretsmanager:us-east-1:{DEFAULT_ACCOUNT_ID}:secret:{name}"
        try:
            secret_value = client.get_secret_value(SecretId=partial_arn)
            print("Successfully retrieved secret.", secret_value)
        except Exception as e:
            print("Error occurred:", e)


if __name__ == "__main__":
    test_partial_arn()
