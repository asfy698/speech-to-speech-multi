#bot_face
import pygame
import math
import random

pygame.init()

# =========================
# SCREEN
# =========================
WIDTH, HEIGHT = 1024, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced Robot Face")

clock = pygame.time.Clock()

# =========================
# COLORS
# =========================
BG = (5, 5, 10)

CYAN = (0, 255, 255)
DARK_CYAN = (0, 120, 120)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# =========================
# EYE SETTINGS
# =========================
LEFT_EYE_X = 340
RIGHT_EYE_X = 684
EYE_Y = 220

EYE_RADIUS = 85
PUPIL_RADIUS = 22

blink_timer = 0
blink_duration = 0
is_blinking = False

# =========================
# MOUTH SETTINGS
# =========================
mouth_mode = "smile"

# =========================
# FUNCTIONS
# =========================
def draw_glow_circle(surface, color, center, radius):
    for i in range(12, 0, -1):
        alpha_surface = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)

        glow_color = (*color, 8)
        pygame.draw.circle(
            alpha_surface,
            glow_color,
            (radius * 2, radius * 2),
            radius + i * 4
        )

        surface.blit(
            alpha_surface,
            (center[0] - radius * 2, center[1] - radius * 2)
        )


def draw_eye(x, y, blink=False):
    if blink:
        pygame.draw.line(
            screen,
            CYAN,
            (x - 70, y),
            (x + 70, y),
            12
        )
        return

    # Glow
    draw_glow_circle(screen, CYAN, (x, y), EYE_RADIUS)

    # Outer eye
    pygame.draw.circle(screen, CYAN, (x, y), EYE_RADIUS, 6)

    # Inner eye
    pygame.draw.circle(screen, BLACK, (x, y), EYE_RADIUS - 5)

    # Animated pupil movement
    t = pygame.time.get_ticks() * 0.002

    pupil_x = x + int(math.sin(t) * 18)
    pupil_y = y + int(math.cos(t * 1.3) * 10)

    pygame.draw.circle(screen, CYAN, (pupil_x, pupil_y), PUPIL_RADIUS)

    # Eye shine
    pygame.draw.circle(screen, WHITE,
                       (pupil_x - 8, pupil_y - 8), 5)


def draw_mouth():
    t = pygame.time.get_ticks() * 0.005

    mouth_width = 260
    mouth_height = 70

    rect = pygame.Rect(
        WIDTH // 2 - mouth_width // 2,
        380,
        mouth_width,
        mouth_height
    )

    # Glow layers
    for i in range(10, 0, -1):
        glow_surface = pygame.Surface((mouth_width + 100, mouth_height + 100), pygame.SRCALPHA)

        pygame.draw.arc(
            glow_surface,
            (*CYAN, 10),
            (50, 50, mouth_width, mouth_height),
            math.radians(180),
            math.radians(360),
            8
        )

        screen.blit(
            glow_surface,
            (rect.x - 50, rect.y - 50)
        )

        pygame.draw.arc(
            screen,
            CYAN,
            rect,
            math.radians(180),
            math.radians(360),
            8
        )


def draw_background_effects():
    # Animated horizontal scanning lines
    for y in range(0, HEIGHT, 6):
        alpha = random.randint(10, 30)

        line_surface = pygame.Surface((WIDTH, 2), pygame.SRCALPHA)
        line_surface.fill((0, 255, 255, alpha))

        screen.blit(line_surface, (0, y))

    # Floating particles
    for i in range(25):
        px = (pygame.time.get_ticks() * 0.05 + i * 40) % WIDTH
        py = (i * 27 + math.sin(i + pygame.time.get_ticks() * 0.001) * 20) % HEIGHT

        pygame.draw.circle(screen, DARK_CYAN, (int(px), int(py)), 2)


# =========================
# MAIN LOOP
# =========================
running = True

while running:

    screen.fill(BG)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Background animation
    draw_background_effects()

    # Random blinking
    blink_timer += 1

    if not is_blinking and blink_timer > random.randint(180, 420):
        is_blinking = True
        blink_duration = 0

    if is_blinking:
        blink_duration += 1

        draw_eye(LEFT_EYE_X, EYE_Y, True)
        draw_eye(RIGHT_EYE_X, EYE_Y, True)

        if blink_duration > 8:
            is_blinking = False
            blink_timer = 0

    else:
        draw_eye(LEFT_EYE_X, EYE_Y)
        draw_eye(RIGHT_EYE_X, EYE_Y)

    # Mouth
    draw_mouth()

    pygame.display.update()
    clock.tick(60)

pygame.quit()
