# Gholamrezadar Dec 2020
# %%
import face_recognition
import numpy as np
from PIL import Image, ImageDraw,ImageFont
from IPython.display import display

# Loading first image and computing encodings
ghd_image = face_recognition.load_image_file("images/ghd1.jpg")
ghd_face_encoding = face_recognition.face_encodings(ghd_image)[0]

# Loading second image and computing encodings
hrad_image = face_recognition.load_image_file("images/hamid2.jpg")
hrad_face_encoding = face_recognition.face_encodings(hrad_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    ghd_face_encoding,
    hrad_face_encoding
]
known_face_names = [
    "Gholamreza",
    "Hamidreza"
]
for i,enc in enumerate(known_face_encodings):
    print(known_face_names[i], "Encoding")
    print(enc)
    print()


# %%
print(len(ghd_face_encoding))


# %%
print("face_distance between ghd and hrad",face_recognition.face_distance([ghd_face_encoding], hrad_face_encoding)[0])


# %%
# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("images/hamid1.jpg")

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
log_message = ""
face_id = 0
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    face_id += 1
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    print("\nface #{}".format(face_id))
    print("leftmost pixel :",left,"px ")
    print(known_face_names)
    print(matches)

    name = "Unknown"

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    print(face_distances)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    print(name)
    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    font = ImageFont.truetype("arial.ttf",15)
    text_width, text_height = draw.textsize(name, font=font)
    draw.rectangle(((left, bottom - text_height - 10+20), (right, bottom+20)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5+20), name, font=font, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
display(pil_image)


# %%
print(pil_image.size[0])


# %%
images_list = ["hamid1","hamid2","hamid3","hamid4","hamid5","hamid and ghd","hamid and seif","hamid and ghd 2","ghd1","ghd2","ghd3"]
for img in images_list:
    # Load an image with an unknown face
    unknown_image = face_recognition.load_image_file("images/{}.jpg".format(img))

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    log_message = ""
    face_id = 0
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_id += 1
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        print("\n\n\nface #{}".format(face_id))
        print("leftmost pixel :",left,"px ")
        print(known_face_names)
        print(matches)

        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        print(face_distances)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        print(name)
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        font = ImageFont.truetype("arial.ttf",15)
        text_width, text_height = draw.textsize(name, font=font)
        draw.rectangle(((left, bottom - text_height - 10+20), (right, bottom+20)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5+20), name, font=font, fill=(255, 255, 255, 255))


    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    display(pil_image)
    pil_image.save("images/results/{}.jpg".format(img))


# %%



