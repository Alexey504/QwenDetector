import torch
import json
import os
from PIL import ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from .model_loader import get_model_and_processor
import tempfile


def create_annotated_image(image, json_data, height, width):
    """Создаёт аннотированное изображение по JSON-ответу от Qwen."""

    try:
        parsed_json_data = json_data.split("```json")[1].split("```")[0]
        bbox_data = json.loads(parsed_json_data)
    except Exception:
        return image, []

    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    labels = []

    for obj in bbox_data:
        bbox = obj.get("bbox_2d") or obj.get("bbox") or obj.get("box") or []
        label = obj.get("label", "")
        labels.append(label)

        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            scale_x = image.width / width
            scale_y = image.height / height
            x1, y1, x2, y2 = [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]

            draw.rectangle([x1, y1, x2, y2], outline="magenta", width=3)

            try:
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_w, text_h = font.getsize(label)

            text_bg = [x1, y1 - text_h - 4, x1 + text_w + 6, y1]
            draw.rectangle(text_bg, fill="magenta")
            draw.text((x1 + 3, y1 - text_h - 2), label, fill="white", font=font)

    return annotated, labels


def generate_visual_answer(image, user_prompt):

    model, processor = get_model_and_processor()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
    try:
        image.save(temp_path)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": temp_path},
                {"type": "text", "text": (
                    f"{user_prompt}\n"
                    "Сначала верни JSON с координатами объектов в формате ```json```, "
                    "затем после JSON добавь короткое текстовое описание сцены."
                )},
            ],
        }]

        if image is None or image.size[0] == 0 or image.size[1] == 0:
            raise ValueError("Пустое или повреждённое изображение")

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            output_text = processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    annotated, _ = create_annotated_image(
        image=image,
        json_data=output_text,
        height=image.height,
        width=image.width,
    )

    # image description
    description = "Описание недоступно"
    if "```json" in output_text:
        try:
            description = output_text.split("```json")[1].split("```")[1].strip()
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return annotated, description
