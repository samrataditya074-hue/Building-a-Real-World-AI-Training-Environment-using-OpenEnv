import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

def main():
    api_key = os.getenv("GROQ_API_KEY", os.getenv("HF_TOKEN"))
    base_url = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    model_name = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Simple local execution of environment for demonstration
    from customer_support_env.environment import SupportEnvironment
    from customer_support_env.models import Action

    env = SupportEnvironment()
    obs = env.reset(seed=42)
    
    print("[START]")
    
    messages = [
        {"role": "system", "content": "You are a customer support agent. Available actions: categorize (return category name), reply (return response text), escalate (return reason). Respond only in JSON: {'action_type': '...', 'content': '...'}"},
        {"role": "user", "content": f"Ticket: {obs.customer_message}. Urgency: {obs.urgency}"}
    ]

    for step in range(1, 6):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            act_type = result.get("action_type", "reply")
            content = result.get("content", "I am helping.")
            
            action = Action(action_type=act_type, content=content)
            obs = env.step(action)
            
            print("[STEP]")
            print(f"Step {step}: action_type={act_type} content={content}")
            print(f"Reward: {obs.reward:.3f}  Done: {obs.done}")
            
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            messages.append({"role": "user", "content": f"Action resulted in reward {obs.reward:.3f}. Done: {obs.done}"})
            
            if obs.done:
                break
        except Exception as e:
            print(f"Error during inference: {e}")
            break

    print("[END]")

if __name__ == "__main__":
    main()
