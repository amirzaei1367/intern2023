import boto3
import botocore
import fire

def main(
	BUCKET_NAME = 'resume-database-intern2023', # replace with your bucket name
	KEY = 'resume_fields.csv' # replace with your object key
):
	s3 = boto3.resource('s3')

	try:
    		s3.Bucket(BUCKET_NAME).download_file(KEY, KEY)
	except botocore.exceptions.ClientError as e:
    		if e.response['Error']['Code'] == "404":
        		print("The object does not exist.")
    		else:
        		raise
if __name__ == "__main__":
	fire.Fire(main)
