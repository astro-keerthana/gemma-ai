"""
app.py - Gradio demo for Sanjeevani Triage.

Run: python app.py
Then open the local URL Gradio prints (add share=True in launch() for a
public link to put in your Kaggle write-up / demo video).
"""

import gradio as gr

from triage.gemma_client import GemmaClient
from triage.triage_engine import run_triage

client = GemmaClient()  # auto-detects ollama / gemini / mock, see gemma_client.py

TIER_COLOR = {
    "EMERGENCY": "🔴",
    "URGENT": "🟠",
    "ROUTINE": "🟡",
    "SELF_CARE": "🟢",
    "NEEDS_MORE_INFO": "🔵",
}


def triage_turn(patient_text, history_state):
    if not patient_text or not patient_text.strip():
        return "Please describe how you're feeling.", history_state, ""

    history_state = history_state or []
    result = run_triage(patient_text, conversation_history=history_state, client=client)
    history_state = history_state + [f"Patient: {patient_text}"]

    icon = TIER_COLOR.get(result.tier.value, "")
    if result.needs_follow_up and result.symptom_record.follow_up_question:
        chat_reply = f"{result.symptom_record.follow_up_question}"
        history_state.append(f"Assistant: {chat_reply}")
    else:
        chat_reply = (
            f"{icon} Urgency: **{result.tier.value}**\n\n{result.rationale}\n\n"
            f"(backend: {result.symptom_record.raw_backend} / {result.symptom_record.raw_model})"
        )
        history_state.append(f"Assistant: [triage complete - {result.tier.value}]")

    return chat_reply, history_state, result.handoff_note


with gr.Blocks(title="Sanjeevani Triage") as demo:
    gr.Markdown(
        "# 🩺 Sanjeevani Triage\n"
        "Multilingual health triage assistant powered by Gemma 4. "
        "Describe symptoms in English, Hindi, Romanized Hindi, or Chhattisgarhi.\n\n"
        "**Prototype — not a substitute for professional medical advice. "
        "In a real emergency, call your local emergency number immediately.**"
    )

    with gr.Row():
        with gr.Column(scale=2):
            patient_input = gr.Textbox(
                label="Describe your symptoms",
                placeholder="e.g. 'Mujhe do din se tez bukhar hai' / 'I have chest pain and shortness of breath'",
                lines=3,
            )
            submit_btn = gr.Button("Submit", variant="primary")
            reply_box = gr.Markdown(label="Response")
        with gr.Column(scale=1):
            handoff_box = gr.Textbox(label="Clinical handoff note", lines=18, interactive=False)

    history_state = gr.State([])

    submit_btn.click(
        triage_turn,
        inputs=[patient_input, history_state],
        outputs=[reply_box, history_state, handoff_box],
    )
    patient_input.submit(
        triage_turn,
        inputs=[patient_input, history_state],
        outputs=[reply_box, history_state, handoff_box],
    )

if __name__ == "__main__":
    demo.launch()
