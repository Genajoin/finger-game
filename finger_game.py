import cv2
import mediapipe as mp
import pygame
import numpy as np
import time

# --- Настройки игры ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TARGET_DURATION = 5  # Сколько секунд показывать задание перед сменой (если не угадал)

# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Учим цифры с помощью рук!")
font_large = pygame.font.SysFont("Arial", 80, bold=True)
font_small = pygame.font.SysFont("Arial", 30)

# Инициализация MediaPipe (старый API)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (50, 50, 255)


class FingerGame:
    """Основной класс игры для распознавания количества пальцев на руке."""
    
    def __init__(self):
        """Инициализация игры: запуск камеры, генерация первого задания."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру")
        self.target_number = np.random.randint(1, 10)  # Случайное число от 1 до 9
        self.user_count = 0
        self.game_state = "ask"  # "ask" (задание), "correct" (правильно), "wrong" (неправильно)
        self.state_start_time = time.time()
    
    def get_finger_count(self, landmarks: list) -> int:
        """
        Определяет, сколько пальцев показано на одной руке.
        
        Логика: Палец считается поднятым, если кончик пальца выше сустава.
        Для большого пальца используется ось X и определение правой/левой руки.
        
        Args:
            landmarks: Список ключевых точек руки от MediaPipe (21 точка)
            
        Returns:
            Количество поднятых пальцев (от 0 до 5)
        """
        count = 0
        fingers = [8, 12, 16, 20]  # Кончики указательного, среднего, безымянного, мизинца
        knuckles = [6, 10, 14, 18]  # Основные суставы (PIP) для этих пальцев
        
        # 1. Проверяем 4 пальца (указательный, средний, безымянный, мизинец)
        # У MediaPipe Y координата уменьшается снизу вверх. 
        # Если tip_y < knuckle_y, значит палец поднят вверх.
        for tip, knuckle in zip(fingers, knuckles):
            if landmarks[tip].y < landmarks[knuckle].y:
                count += 1

        # 2. Проверяем большой палец
        # Для большого пальца нужна проверка по X координате, так как он в другой плоскости.
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_mcp = landmarks[5]  # Сустав большого пальца у основания указательного
        
        # Определяем руку (левая или правая) по положению wrist и middle_finger
        wrist = landmarks[0]
        middle_finger = landmarks[12]
        
        # Если wrist.x < middle_finger.x, то это правая рука (wrist левее среднего пальца)
        is_right_hand = wrist.x < middle_finger.x
        
        # Если правая рука: большой палец должен быть левее (x меньше) IP сустава
        # Если левая рука: большой палец должен быть правее (x больше) IP сустава
        if is_right_hand:
            if thumb_tip.x < thumb_ip.x:
                count += 1
        else:
            if thumb_tip.x > thumb_ip.x:
                count += 1
                
        return count

    def draw_text_with_outline(self, text_string: str, center_x: int, center_y: int, font, text_color=BLACK, outline_color=WHITE, outline_width: int = 2):
        """Рисует текст с белой обводкой для лучшей видимости."""
        # Рисуем обводку (белый текст с небольшим смещением)
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    outline_surface = font.render(text_string, True, outline_color)
                    offset_rect = outline_surface.get_rect(center=(center_x + dx, center_y + dy))
                    screen.blit(outline_surface, offset_rect)
        
        # Основной текст
        text_surface = font.render(text_string, True, text_color)
        rect = text_surface.get_rect(center=(center_x, center_y))
        screen.blit(text_surface, rect)

    def draw_on_screen(self, frame, landmarks=None):
        """Отрисовка интерфейса поверх видео."""
        # Переводим кадр из BGR (OpenCV) в RGB для Pygame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = pygame.surfarray.make_surface(frame)
        frame = pygame.transform.rotate(frame, -90)  # Поворот по часовой стрелке на 90 градусов
        
        # Масштабируем кадр под размер окна
        frame = pygame.transform.scale(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
        screen.blit(frame, (0, 0))

        # Отрисовка текстовых подсказок
        if self.game_state == "ask":
            # Текст задания с белой обводкой
            self.draw_text_with_outline(f"Покажи: {self.target_number}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3, font_large, BLACK)
            
            # Показываем распознанное количество (для наглядности)
            count_text = font_small.render(f"Ты показываешь: {self.user_count}", True, BLUE)
            count_rect = count_text.get_rect(center=(SCREEN_WIDTH - 150, 50))
            screen.blit(count_text, count_rect)
            
        elif self.game_state == "correct":
            # Анимация правильного ответа
            text = font_large.render("ВЕРНО!", True, GREEN)
            rect = text.get_rect(center=((SCREEN_WIDTH // 2) - 100, SCREEN_HEIGHT // 2))
            screen.blit(text, rect)
            
            # Подсказка следующего числа
            hint = font_small.render("Нажми 'Пробел' для следующего числа", True, BLACK)
            hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60))
            screen.blit(hint, hint_rect)
            
        elif self.game_state == "timeout":
            text = font_large.render("ВРЕЯ ВЫШЛО! Попробуй ещё раз.", True, RED)
            rect = text.get_rect(center=(SCREEN_WIDTH // 2, 100))
            screen.blit(text, rect)

        pygame.display.flip()

    def run(self):
        """Запуск основного цикла игры."""
        clock = pygame.time.Clock()
        running = True

        while running:
            # 1. Захват кадра
            success, image = self.cap.read()
            if not success:
                print("Не удалось получить кадр с камеры")
                break
            
            # Инвертируем по горизонтали (чтобы было привычно)
            #image = cv2.flip(image, 1)  # Отключено для теста
            
            # Конвертация в RGB для MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            self.user_count = 0
            
            # 2. Логика игры
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Рисуем скелет руки
                    mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Считаем пальцы
                    self.user_count += self.get_finger_count(hand_landmarks.landmark)
                    
                    # Если мы в режиме ожидания (задали число, ждем ответа)
                    if self.game_state == "ask":
                        if self.user_count == self.target_number:
                            self.game_state = "correct"
                            self.state_start_time = time.time()
            else:
                # Если руки нет
                if self.game_state != "correct":
                    self.user_count = "?"

            # 3. Управление временем
            current_time = time.time()
            if self.game_state == "correct":
                if current_time - self.state_start_time > 3:  # Ждем 3 секунды
                    self.target_number = np.random.randint(1, 11)  # Новое число (1..10)
                    self.game_state = "ask"
                    self.state_start_time = current_time

            # 4. Обработка событий (клавиатура)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and self.game_state == "correct":
                        # Принудительная смена числа (если ребенок не хочет ждать)
                        self.target_number = np.random.randint(1, 11)
                        self.game_state = "ask"
                        self.state_start_time = current_time

            # 5. Отрисовка
            self.draw_on_screen(image, None)

            clock.tick(30)  # Ограничение FPS

        self.cap.release()
        pygame.quit()


if __name__ == "__main__":
    try:
        game = FingerGame()
        game.run()
    except KeyboardInterrupt:
        print("\nИгра прервана пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        if 'game' in locals() and hasattr(game, 'cap'):
            game.cap.release()
        pygame.quit()
