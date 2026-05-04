import asyncio
import aiohttp
from time import perf_counter


def normalize_messages(prompt_or_messages):
    if isinstance(prompt_or_messages, str):
        return [{"role": "user", "content": prompt_or_messages}]
    if isinstance(prompt_or_messages, list):
        return prompt_or_messages
    raise TypeError("prompt must be a string or an OpenAI-compatible messages list")


async def create_completion(session, prompt, model="gpt-4o-mini", return_usage=False):
    API_KEY = ''
    BASE_URL = ""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    retry = 0
    last_error = None
    while retry < 5:
        try:
            started_at = perf_counter()
            async with session.post(url=f"{BASE_URL}chat/completions",
                json={
                    "model": model,
                    "max_tokens":4000,
                    "temperature": 0,
                    # "frequency_penalty": 0.05,
                    # "presence_penalty": 0.0,
                    # "top_p": 1,
                    "messages": normalize_messages(prompt),
                },
                headers=headers
            ) as response:
                api_latency_s = perf_counter() - started_at
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    print(content)
                    print('-----------------------')
                    if return_usage:
                        return {
                            "content": content,
                            "usage": result.get("usage") or {},
                            "api_latency_s": api_latency_s,
                        }
                    return content
                else:
                    last_error = f"请求失败，状态码: {response.status}"
                    print(last_error)
                    retry += 1
                    await asyncio.sleep(1)
        except Exception as e:
            last_error = e
            retry += 1
            await asyncio.sleep(1)
            print(f"Error: {e}", flush=True)
    if return_usage:
        return {"content": "", "usage": {}, "api_latency_s": 0, "error": str(last_error)}
    return None

async def run_async(prompts, model="gpt-4o-mini", return_usage=False, max_concurrency=4):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_create_completion(session, prompt):
        async with semaphore:
            return await create_completion(session, prompt, model, return_usage=return_usage)

    async with aiohttp.ClientSession() as session:
        tasks = [limited_create_completion(session, prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
    return responses

if __name__ == "__main__":
    prompts = ['who are you?','what you like?']
    responses = asyncio.run(run_async(prompts))
    print('--------')
    print(responses)
