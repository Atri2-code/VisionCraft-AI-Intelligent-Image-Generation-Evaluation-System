def enhance_prompt(prompt):
    return f"high quality, ultra detailed, professional design, {prompt}"

def generate_feedback(scores):
    if max(scores) - min(scores) < 0.1:
        return "Images are similar. Try refining the prompt."
    return "Top image aligns best with the prompt."
