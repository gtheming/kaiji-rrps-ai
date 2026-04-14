import pygame

_surface = None
_clock = None

CELL_SIZE = 60
PADDING = 40

# Player type colors
TYPE_COLORS = {
    "R": (220, 80, 80),  # Rock  - red
    "P": (80, 140, 220),  # Paper - blue
    "S": (80, 200, 120),  # Scissors - green
}
DEAD_COLOR = (60, 60, 70)
BG_COLOR = (15, 15, 20)
GRID_COLOR = (40, 48, 68)
TEXT_COLOR = (220, 225, 235)


def init(grid_rows, grid_cols):
    global _surface, _clock
    if not pygame.get_init():
        pygame.init()
    width = grid_cols * CELL_SIZE + PADDING * 2
    height = grid_rows * CELL_SIZE + PADDING * 2
    _surface = pygame.display.set_mode((width, height), pygame.SHOWN)
    pygame.display.set_caption("RPS Arena — Grid View")
    _clock = pygame.time.Clock()


def draw_to(surface, x_off, y_off, alive_player_dict, grid_rows, grid_cols):
    """Draw the grid onto *surface* at pixel offset (x_off, y_off)."""
    font = pygame.font.SysFont("monospace", 13)
    id_font = pygame.font.SysFont("monospace", 9)

    pos_map = {
        (int(p["position"][0]), int(p["position"][1])): (pid, p)
        for pid, p in alive_player_dict.items()
    }

    for r in range(grid_rows):
        for c in range(grid_cols):
            x = x_off + PADDING + c * CELL_SIZE
            y = y_off + PADDING + r * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            if (r, c) in pos_map:
                pid, info = pos_map[(r, c)]
                ptype = info.get("type", "?")
                color = TYPE_COLORS.get(ptype, (180, 180, 180))
                pygame.draw.rect(surface, color, rect, border_radius=8)
                pygame.draw.rect(surface, GRID_COLOR, rect, 1, border_radius=8)

                label = font.render(ptype, True, (255, 255, 255))
                surface.blit(
                    label,
                    (
                        rect.centerx - label.get_width() // 2,
                        rect.centery - label.get_height() // 2,
                    ),
                )

                id_surf = id_font.render(str(pid), True, (200, 200, 200))
                surface.blit(id_surf, (rect.x + 3, rect.y + 3))
            else:
                pygame.draw.rect(surface, BG_COLOR, rect)
                pygame.draw.rect(surface, GRID_COLOR, rect, 1)


def render(alive_player_dict, grid_rows, grid_cols):
    """Standalone render — only used when grid_view owns its own window."""
    if _surface is None:
        return

    pygame.event.pump()
    _surface.fill(BG_COLOR)
    draw_to(_surface, 0, 0, alive_player_dict, grid_rows, grid_cols)
    pygame.display.flip()
    _clock.tick(60)
