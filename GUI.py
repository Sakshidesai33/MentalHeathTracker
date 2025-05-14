import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import random
import requests
import pygame
import os
import math
import google.generativeai as genai

# Configure Gemini API Key
genai.configure(api_key="AIzaSyB5BrkjWdKU4fJaogTHwFhXjEwGd6lGq3I")  # Replace with your actual Gemini API key

# Emotion to Gemini prompt mapping
emotion_to_prompt = {
    "sad": "Give a happy, uplifting motivational quote stated by a famous personality to counter sadness.",
    "joy": "Give an encouraging quote from a famous person to sustain joy and motivation.",
    "anger": "Give a calming, peaceful quote from a well-known figure to counteract anger.",
    "fear": "Give a strong, confidence-boosting quote by a famous person to overcome fear.",
    "surprise": "Give a grounding quote from a famous person to bring calm to someone surprised or overwhelmed.",
    "love": "Give a motivational quote focused on self-love, by a famous personality."
}

def generate_quote(emotion):
    prompt = emotion_to_prompt.get(emotion.lower(), "Give a motivational quote by a famous person.")
    try:
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating quote: {e}"

# Initialize pygame mixer for music
pygame.mixer.init()
music_path = r"C:\Users\saksh\Desktop\Minor Project\nature-calming-310735[1].mp3"

def play_music_loop():
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    if os.path.exists(music_path):
        try:
            pygame.mixer.music.load(music_path)
            pygame.mixer.music.set_volume(0.5)
            pygame.mixer.music.play(-1)
        except Exception as e:
            print("Error playing music:", e)
    else:
        print("Music file not found!")

def stop_music():
    pygame.mixer.music.stop()

# Load trained LSTM model and tokenizer
model = tf.keras.models.load_model("lstm_sentiment_model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)
with open("label_encoder.pkl", "rb") as handle:
    label_encoder = pickle.load(handle)

mood_boosting_emojis = [
    "🌟", "💖", "💪", "😊", "🌈", "✨", "🔥", "🎉", "🌞", "🌻", "😃", "🎶",
    "😇", "💫", "🥳", "🤗", "🕊️", "🌺", "🚀", "🎈", "🏆", "❤️", "🤩", "🙌"
]

# 🎮 Flower Bloom Game
def launch_flower_game():
    pygame.init()
    play_music_loop()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("🌸 Flower Bloom Animation")
    clock = pygame.time.Clock()
    FPS = 60

    BACKGROUND = (255, 255, 255)
    CENTER_COLOR = (255, 204, 255)
    PETAL_COLORS = [
        (255, 105, 180), (135, 206, 250), (255, 182, 193),
        (144, 238, 144), (255, 255, 102)
    ]

    class Flower:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.radius = 0
            self.max_radius = random.randint(40, 70)
            self.growth_speed = 2
            self.petals = random.randint(5, 8)
            self.color = random.choice(PETAL_COLORS)
            self.center_size = 5

        def update(self):
            if self.radius < self.max_radius:
                self.radius += self.growth_speed

        def draw(self, screen):
            angle_step = 2 * math.pi / self.petals
            for i in range(self.petals):
                angle = i * angle_step
                dx = math.cos(angle) * self.radius
                dy = math.sin(angle) * self.radius
                petal_center = (int(self.x + dx), int(self.y + dy))
                pygame.draw.circle(screen, self.color, petal_center, int(self.radius / 3))
            pygame.draw.circle(screen, CENTER_COLOR, (self.x, self.y), self.center_size)

    flowers = []
    running = True
    while running:
        screen.fill(BACKGROUND)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if 10 <= x <= 90 and 10 <= y <= 50:
                    running = False
                else:
                    flowers.append(Flower(x, y))

        for flower in flowers:
            flower.update()
            flower.draw(screen)

        pygame.draw.rect(screen, (255, 99, 71), (10, 10, 80, 40))
        font = pygame.font.SysFont(None, 24)
        text = font.render("Exit", True, (255, 255, 255))
        screen.blit(text, (30, 20))

        pygame.display.update()
        clock.tick(FPS)

    stop_music()
    pygame.quit()
    pygame.mixer.init()
    launch_game_menu()

# 🎮 Bubble Pop Game
def launch_bubble_game():
    pygame.init()
    play_music_loop()
    WIDTH, HEIGHT = 600, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("🪧 Bubble Pop Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 28)
    score = 0

    class Bubble:
        def __init__(self):
            self.radius = random.randint(20, 40)
            self.x = random.randint(self.radius, WIDTH - self.radius)
            self.y = HEIGHT + self.radius
            self.color = random.choice([(135,206,250), (173,216,230), (0,191,255), (255,182,193), (144,238,144)])
            self.speed = random.uniform(1.5, 3.5)
            self.popped = False

        def move(self):
            self.y -= self.speed

        def draw(self):
            if not self.popped:
                pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius, 3)

        def is_clicked(self, pos):
            dx = self.x - pos[0]
            dy = self.y - pos[1]
            return math.sqrt(dx*dx + dy*dy) <= self.radius

    BUTTON_RECT = pygame.Rect(WIDTH - 130, 10, 110, 40)
    BUTTON_COLOR = (220, 20, 60)
    BUTTON_TEXT = font.render("Exit", True, (255, 255, 255))

    bubbles = []
    spawn_timer = 0
    running = True

    while running:
        screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if BUTTON_RECT.collidepoint(mouse_pos):
                    running = False
                for bubble in bubbles:
                    if bubble.is_clicked(mouse_pos) and not bubble.popped:
                        bubble.popped = True
                        score += 1

        spawn_timer += 1
        if spawn_timer > 30:
            bubbles.append(Bubble())
            spawn_timer = 0

        for bubble in bubbles[:]:
            bubble.move()
            bubble.draw()
            if bubble.y < -bubble.radius or bubble.popped:
                bubbles.remove(bubble)

        score_text = font.render(f"Score: {score}", True, (30, 144, 255))
        screen.blit(score_text, (10, 10))

        pygame.draw.rect(screen, BUTTON_COLOR, BUTTON_RECT, border_radius=8)
        screen.blit(BUTTON_TEXT, (BUTTON_RECT.x + 25, BUTTON_RECT.y + 7))

        pygame.display.update()
        clock.tick(60)

    stop_music()
    pygame.quit()
    pygame.mixer.init()
    launch_game_menu()

# Game Selection Menu
def launch_game_menu():
    menu = tk.Toplevel()
    menu.title("Choose a Game")
    menu.geometry("400x200")
    menu.config(bg="#e6e6fa")

    label = tk.Label(menu, text="Feeling off? Want to lift your mood?\nChoose a game below:", font=("Arial", 12, "bold"), bg="#e6e6fa", fg="#6a0dad")
    label.pack(pady=20)

    flower_btn = tk.Button(menu, text="🌸 Flower Bloom Game", font=("Arial", 12, "bold"), bg="#ffb6c1", fg="white", command=lambda: [menu.withdraw(), launch_flower_game()])
    flower_btn.pack(pady=5)

    bubble_btn = tk.Button(menu, text="🪧 Bubble Pop Game", font=("Arial", 12, "bold"), bg="#87ceeb", fg="white", command=lambda: [menu.withdraw(), launch_bubble_game()])
    bubble_btn.pack(pady=5)

# Emotion Prediction
def predict_emotion():
    user_input = text_entry.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showerror("Error", "Please enter some text")
        return

    sequence = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequence, maxlen=120, padding="post", truncating="post")
    prediction = np.argmax(model.predict(padded), axis=-1)
    emotion = label_encoder.inverse_transform(prediction)[0]

    result_label.config(text=f"Predicted Emotion: {emotion}", fg="#ffffff", bg="#d8bfd8")
    result_label.update_idletasks()

    calming_message = (
        "Emotions are valid—here’s something soothing for your soul... 🌈🎵\n"
        "\nAnd here's a motivational quote to lift you up:\n\n" +
        generate_quote(emotion)
    )
    quote_label.config(text=calming_message)
    emoji_label.config(text=random.choice(mood_boosting_emojis), font=("Arial", 72), fg="#ffcc00")

    if emotion.lower() in ["sad", "anger"]:
        play_music_loop()
        launch_game_menu()
    else:
        stop_music()

# GUI Setup
root = tk.Tk()
root.title("Emotion Detector")
root.geometry("700x750")
root.config(bg="#d8bfd8")

frame = tk.Frame(root, bg="#ffffff", padx=20, pady=20, relief="ridge", borderwidth=8)
frame.pack(pady=20)

title_label = tk.Label(frame, text="🌈 Emotion Detector 🌈", font=("Arial", 22, "bold"), bg="#ffffff", fg="#6a0dad")
title_label.pack(pady=10)

text_entry = tk.Text(frame, height=5, width=50, font=("Arial", 12), relief="solid", borderwidth=3, bg="#fff8dc", fg="#333333", padx=10, pady=10)
text_entry.pack(pady=10)

analyze_button = tk.Button(frame, text="✨ Analyze Emotion ✨", command=predict_emotion, font=("Arial", 14, "bold"), bg="#9370db", fg="#ffffff", padx=10, pady=5, borderwidth=3, relief="raised")
analyze_button.pack(pady=10)

result_label = tk.Label(frame, text="", font=("Arial", 16, "bold"), width=40, bg="#d8bfd8", fg="#ffffff", padx=10, pady=10, relief="solid", borderwidth=3)
result_label.pack(pady=10)

quote_label = tk.Label(frame, text="", font=("Arial", 14, "italic"), wraplength=500, bg="#ffffff", fg="#6a0dad", padx=10, pady=10)
quote_label.pack(pady=10)

emoji_label = tk.Label(frame, text="", font=("Arial", 72), bg="#ffffff", fg="#ffcc00")
emoji_label.pack(pady=10)

root.mainloop()
