import json
import boto3

def lambda_handler(event, context):
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    
    # Extract input text from incoming test event or use sample text
    input_text = event.get("input_text", "URGENT: Client wants the updated project scope by 3 PM today. Also don't forget to grab lunch.")
    
    system_prompt = """
    You are an autonomous Task Triage AI Agent. 
    Analyze the user input text and generate a structured output containing:
    1. Urgency Rank: [Low / Medium / High / Critical]
    2. Category: [Work / Personal / Admin / Urgent]
    3. Action Items: (Bullet points)
    4. Suggested Reply: (A clear, polite response draft)
    """

    prompt_payload = {
        "inferenceConfig": {"maxTokens": 500, "temperature": 0.3},
        "messages": [
            {"role": "user", "content": [{"text": f"{system_prompt}\n\nInput Text: {input_text}"}]}
        ]
    }

    try:
        response = bedrock.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps(prompt_payload)
        )
        result = json.loads(response['body'].read())
        agent_output = result['output']['message']['content'][0]['text']
        
        return {
            'statusCode': 200,
            'body': json.dumps({'triage_result': agent_output})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
