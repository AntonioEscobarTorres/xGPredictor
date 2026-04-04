import ast
import numpy as np
import pandas as pd


def normalize_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Garantir que 'location' seja uma lista de verdade (converte string se necessário)
    def ensure_list(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                return [np.nan, np.nan]
        return x

    # 2. Extrair x e y com segurança
    locs = df['location'].apply(ensure_list).tolist()
    
    # Criar colunas temporárias para o cálculo
    loc_df = pd.DataFrame(locs, columns=['location_x', 'location_y'], index=df.index)
    df['location_x'] = loc_df['location_x']
    df['location_y'] = loc_df['location_y']
    
    # 3. Resto da sua lógica de normalização (Inverter campo se x < 60)
    df['x_norm'] = df['location_x']
    df['y_norm'] = df['location_y']
    
    mask = df['location_x'] < 60
    df.loc[mask, 'x_norm'] = 120 - df.loc[mask, 'location_x']
    df.loc[mask, 'y_norm'] = 80 - df.loc[mask, 'location_y']
    
    return df


def check_inside_box(row: pd.Series) -> bool:
    x, y = row['x_norm'], row['y_norm']
    return x >= 102 and 18 <= y <= 62


def check_inverted(row: pd.Series) -> str:
    side = 'left' if row['y_norm'] < 40 else 'right'
    foot = row['shot_body_part']
    if (side == 'left' and foot == 'Right Foot') or (side == 'right' and foot == 'Left Foot'):
        return 'Inverted'
    if (side == 'left' and foot == 'Left Foot') or (side == 'right' and foot == 'Right Foot'):
        return 'Natural'
    return 'Other'


def get_situation(row: pd.Series) -> str:
    area = 'Inside' if row['is_inside_box'] else 'Outside'
    pressure = 'Pressure' if row['under_pressure'] else 'No Pressure'
    return f"{area} Box / {pressure}"


def distance_to_goal(row: pd.Series) -> float:
    x_goal, y_goal = 120, 40
    dx = x_goal - row['x_norm']
    dy = y_goal - row['y_norm']
    return np.sqrt(dx ** 2 + dy ** 2)


def shot_angle(row: pd.Series) -> float:
    x_goal = 120
    y_post1 = 36
    y_post2 = 44
    dx = x_goal - row['x_norm']
    dy1 = y_post1 - row['y_norm']
    dy2 = y_post2 - row['y_norm']
    angle = np.abs(np.arctan2(dy2, dx) - np.arctan2(dy1, dx))
    return np.degrees(angle)



def _parse_freeze_frame(ff):
    if ff is None or (isinstance(ff, float) and np.isnan(ff)):
        return []
    if isinstance(ff, str):
        ff = ff.strip()
        if not ff:
            return []
        try:
            return ast.literal_eval(ff)
        except (ValueError, SyntaxError):
            return []
    if isinstance(ff, list):
        return ff
    return []


def count_blockers(row: pd.Series) -> int:
    frame = _parse_freeze_frame(row['shot_freeze_frame'])
    if not frame:
        return 0
    bx, by = row['x_norm'], row['y_norm']
    angle_post1 = np.arctan2(36 - by, 120 - bx)
    angle_post2 = np.arctan2(44 - by, 120 - bx)
    angle_min, angle_max = sorted((angle_post1, angle_post2))
    blockers = 0
    for player in frame:
        if player.get('teammate', True):
            continue
        loc = player.get('location')
        if not loc or len(loc) < 2:
            continue
        px, py = loc[0], loc[1]
        if px <= bx:
            continue
        angle_player = np.arctan2(py - by, px - bx)
        if angle_min <= angle_player <= angle_max:
            blockers += 1
    return blockers


