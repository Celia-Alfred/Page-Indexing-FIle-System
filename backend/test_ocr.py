
from services.ocr_service import extract_text_from_image


image_path = "sample_page.png.jpeg"

result = extract_text_from_image(image_path)


print("Extracted text:\n")
print(result["text"])

print("\nTotal lines:", result["line_count"])

print("\nBounding boxes and confidence:\n")

for line in result["lines"]:
    print(line)

print("\nCoordinate system:", result["coordinate_system"])
