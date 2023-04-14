import openai

openai.api_key = "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"
prompt = "who are u and how are u?"
response = openai.Completion.create(
    engine = "text-davinci-003",
    prompt = prompt,
    timeout = 20,
    # max_token = 1000,
)

print(response.choices[0].text)