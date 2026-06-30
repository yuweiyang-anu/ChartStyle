"""Image-generation (inference) prompt for the style-transfer models.

A single, shared instruction is passed verbatim to every image-editing backend
(Qwen-Image-Edit, OpenAI gpt-image-*, Gemini / Nano Banana Pro). Keeping it
here, separate from the model code, lets the exact wording used in the paper be
audited and reused without touching the inference logic.

The two images are always provided in the order (style reference, content image).
Do NOT reword this string — the wording is part of the experimental setup.
"""

STYLE_TRANSFER_PROMPT = (
    "I'm providing you two images. The first is a style reference image, and the "
    "second is a content image. The style image is a drawing, i want you to redraw "
    "the content image in the exact same drawing style. I want you to edit the content "
    "image slightly to have exactly the same structure, maintain any text and adjust "
    "the lighting, colour, and style to match the style reference image."
)
