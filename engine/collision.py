"""
Fonctions de collision balle ↔ raquette / filet / table
"""

import pygame
import numpy as np
from config import RESTITUTION, TABLE_Y, VPX_FRAME_MAX, WIDTH, FPS

# a = 0.35 pour de la mousse et 0.22 pour la table
# v0 = 200 m/s pour la mousse et 250 pour la table
def restitution_verticale(vy, v0, a, ey_min=0.85):
    """
    Calcule ey (coeff. de restitution vertical) en fonction de la vitesse verticale vy.

    Params:
      vy      : float, vitesse verticale (m/s) — signe ignoré (on utilise |vy|)
      v0      : float, vitesse de référence (m/s) — pour laquelle ey doit atteindre ey_min
      a       : float, exposant contrôlant la pente (plus a grand → moins de restitution)
      ey_min  : float, valeur minimale de ey (par défaut 0.85)

    Retour :
      ey : float dans [ey_min, 1]
    """
    # normalisation et clamp
    x = abs(vy) / v0
    if x >= 1.0:
        return float(ey_min)

    # noyau : (1 - x)^a décroît quand x augmente ; a plus grand → valeur plus petite
    ey = ey_min + (1.0 - ey_min) * (1.0 - x) ** a

    # sécurité numérique
    if ey < ey_min:
        ey = ey_min
    if ey > 1.0:
        ey = 1.0
    return float(ey)


# pour le moment on prend la même que y
# a = 0.35 pour de la mousse et 0.22 pour la table
# v0 = 200 m/s pour la mousse et 250 pour la table
def restitution_tangentielle(vx, v0, a, ex_min=0.85):
    """
    Calcule le coefficient de restitution tangentielle ex selon la formule empirique :
        ex = 1 - (vx / vx0)**a

    Paramètres :
    ------------
    vx : float
        Vitesse tangentielle (horizontale) avant impact (m/s)
    vx0 : float
        Vitesse caractéristique liée à la friction (m/s)
    a : float
        Exposant empirique lié au type de surface

    Retour :
    --------
    ex : float
        Coefficient de restitution tangentielle (sans unité)
    """

    # normalisation et clamp
    z = abs(vx) / v0
    if z >= 1.0:
        return float(ex_min)

    # noyau : (1 - x)^a décroît quand x augmente ; a plus grand → valeur plus petite
    ex = ex_min + (1.0 - ex_min) * (1.0 - z) ** a

    # sécurité numérique
    if ex < ex_min:
        ex = ex_min
    if ex > 1.0:
        ex = 1.0
    return ex


def adjust_spin_for_corner(angular_speed, ratio, is_left_corner=True):
    """
    Ajuste la vitesse angulaire selon le coin et le ratio.
    - Si la spin est déjà dans le bon sens → augmente avec ratio
    - Sinon → tend vers le bon signe progressivement
    """
    # Coin gauche : bon sens = négatif
    if is_left_corner:
        if angular_speed < 0:
            return angular_speed * (1 + 0.5 * ratio)  # augmente
        else:
            return angular_speed * (1 - 0.8 * ratio) - 0.8 * ratio * abs(angular_speed)  # tend vers négatif
    # Coin droit : bon sens = positif
    else:
        if angular_speed > 0:
            return angular_speed * (1 + 0.5 * ratio)  # augmente
        else:
            return angular_speed * (1 - 0.8 * ratio) + 0.8 * ratio * abs(angular_speed)  # tend vers positif

def reduction_speed(vx, vy, est_mousse):
    if est_mousse:
        vx *= restitution_tangentielle(vx, VPX_FRAME_MAX, 0.35)
        vy *= restitution_verticale(vy, VPX_FRAME_MAX, 0.35)
    else:
        vx *= restitution_tangentielle(vx, VPX_FRAME_MAX*1.2, 0.22)
        vy *= restitution_verticale(vy, VPX_FRAME_MAX*1.2, 0.22)
    return vx, vy


def contact_cercle_rectangle(ball_center_x, ball_center_y, radius,
                             rect_x, rect_y, width, height, angle_deg, screen=None, est_mousse=False):
    """
    Système : 0° = Haut, Rotation Horaire.
    Repère : Pygame (Y+ vers le bas).
    """

    # 0) Setup
    half_w = width / 2.0
    half_h = height / 2.0
    rect_cx = rect_x + half_w
    rect_cy = rect_y + half_h

    # 1) Angle en radians
    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 2) Translation
    dx = ball_center_x - rect_cx
    dy = ball_center_y - rect_cy

    # 3) Rotation inverse (pour passer du monde au repère local du rectangle)
    # Matrice de rotation horaire inverse : 
    # rx =  dx * cos + dy * sin
    # ry = -dx * sin + dy * cos
    rx = dx * cos_t + dy * sin_t
    ry = -dx * sin_t + dy * cos_t

    # 4) Trouver le point le plus proche sur le rectangle local (aligné sur les axes)
    closest_x = np.clip(rx, -half_w, half_w)
    closest_y = np.clip(ry, -half_h, half_h)

    # 5) Test de collision
    dist_x = rx - closest_x
    dist_y = ry - closest_y
    distance_sq = dist_x*dist_x + dist_y*dist_y
    hit = distance_sq <= radius*radius

    # Initialisation variables retour
    contact_local = (closest_x, closest_y)
    face = None
    corner_ratio = None
    n_local = (0.0, 0.0)

    # 6) & 7) Classification de la face et normale locale
    outside_x = (rx < -half_w) or (rx > half_w)
    outside_y = (ry < -half_h) or (ry > half_h)

    if outside_x and not outside_y:
        face = 'gauche' if rx < -half_w else 'droite'
        n_local = (-1.0, 0.0) if rx < -half_w else (1.0, 0.0)
    elif outside_y and not outside_x:
        face = 'haut' if ry < -half_h else 'bas'
        n_local = (0.0, -1.0) if ry < -half_h else (0.0, 1.0)
    elif (not outside_x) and (not outside_y):
        # Cas où le centre est à l'intérieur
        dist_to_right = half_w - rx
        dist_to_left  = rx + half_w
        dist_to_top   = ry + half_h
        dist_to_bottom= half_h - ry
        min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        if min_dist == dist_to_left:   face = 'gauche'; n_local = (-1.0, 0.0)
        elif min_dist == dist_to_right: face = 'droite'; n_local = (1.0, 0.0)
        elif min_dist == dist_to_top:   face = 'haut';   n_local = (0.0, -1.0)
        else:                           face = 'bas';    n_local = (0.0, 1.0)
    else:
        # Coins
        left = rx < -half_w
        top = ry < -half_h
        face = 'corner_' + ('h' if top else 'b') + ('g' if left else 'd')
        # Ratio pour l'effet
        dist_edge_x = abs(rx) - half_w
        corner_ratio = min(1.0, abs(dist_edge_x) / radius)
        
        nx_local = rx - closest_x
        ny_local = ry - closest_y
        length = np.hypot(nx_local, ny_local)
        n_local = (nx_local/length, ny_local/length) if length != 0 else (0,0)

    # 8) Projection du point de contact : Local -> Monde
    # Matrice de rotation horaire : 
    # wx =  rx * cos - ry * sin
    # wy =  rx * sin + ry * cos
    contact_world_x = closest_x * cos_t - closest_y * sin_t + rect_cx
    contact_world_y = closest_x * sin_t + closest_y * cos_t + rect_cy
    contact_world = (contact_world_x, contact_world_y)

    # 9) Projection de la normale : Local -> Monde
    # On utilise LA MÊME matrice que l'étape 8
    nx_world = n_local[0] * cos_t - n_local[1] * sin_t
    ny_world = n_local[0] * sin_t + n_local[1] * cos_t
    
    # Normalisation de sécurité
    norm = np.hypot(nx_world, ny_world)
    if norm != 0:
        nx_world /= norm
        ny_world /= norm
    normal_world = (nx_world, ny_world)

    # 10) Tangente
    # Pour un rebond, la tangente est souvent utile pour le spin
    tangent_world = (-ny_world, nx_world)

    # 11) Dessin debug
    if screen is not None:
        corners_local = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        corners_world = []
        for cx, cy in corners_local:
            wx = cx * cos_t - cy * sin_t + rect_cx
            wy = cx * sin_t + cy * cos_t + rect_cy
            corners_world.append((int(wx), int(wy)))
        pygame.draw.polygon(screen, (0, 255, 0), corners_world, 2)
        if hit:
            pygame.draw.circle(screen, (0, 0, 255), (int(contact_world[0]), int(contact_world[1])), 4)
            # Dessin de la normale en rouge
            pygame.draw.line(screen, (255, 0, 0), contact_world, 
                             (contact_world[0] + nx_world * 20, contact_world[1] + ny_world * 20), 2)

    return hit, contact_world, normal_world, tangent_world, face, corner_ratio


# a=0.35 pour la mousse et 0.22 pour la table
def check_rect_collision(ball, rectangle, est_mousse, est_table, a, spin_factor=0.2, screen=None):
    """
    Gestion du rebond sur un rectangle incluant coins gauche/droite avec :
    - ratio basé sur le bord pour un effet plus marqué
    - spin ajusté en fonction du ratio et de la vitesse de base
    """
    ball_center_x = ball.pos[0]
    ball_center_y = ball.pos[1]

    signe_x = (ball.angular_speed) >= 0 # True si spin positif
    signe_y = (ball.angular_speed * ball.vel[0] >= 0)

    if est_mousse:
        _, _, vel_x, vel_y, angle, width, height = rectangle.get_info()
        # Utiliser la vitesse lissée au lieu de la vitesse instantanée pour le transfert
        if hasattr(rectangle, 'smoothed_vel'):
            vel_x, vel_y = rectangle.smoothed_vel[0], rectangle.smoothed_vel[1]
        x, y, _, _ = rectangle.get_rect()
    elif est_table:
        x, y, width, height = rectangle.get_rect()
        vel_x, vel_y = 0, 0
        angle = 0 # on ajoutera 90 après
    else: # filet
        x, y, width, height = rectangle.get_rect()
        vel_x, vel_y = 0, 0
        angle = 0 # on ajoutera 90 après

    hit, contact, normal, tangent, face, corner_ratio = contact_cercle_rectangle(
        ball_center_x, ball_center_y, ball.radius,
        x, y, width, height, angle, screen, est_mousse
    )

    # hit              # bool - Y a-t-il collision? (cercle intersecte rectangle)
    # contact          # (x, y) - Point de contact approximatif (coords monde)
    # normal           # (nx, ny) - Vecteur normal UNITAIRE pointant vers l'extérieur
    #                 #           (perpendiculaire à la surface)
    # tangent          # (tx, ty) - Vecteur tangent UNITAIRE (direction de la surface)
    # face             # str - Quel côté du rectangle? 
    #                 #       'haut', 'bas', 'gauche', 'droite', 'corner_hg', 'corner_hd', 'corner_bg', 'corner_bd'
    # corner_ratio     # float [0, 1] - Distance normalisée au coin (0=centre face, 1=coin exact)

    if not hit:
        return

    
    # if est_mousse:
    #     print("Avant collision:")
    #     print(f"{angle=}")
    #     print(f"raquette {vel_x=}, {vel_y=}")
    #     # Afficher la vitesse angulaire lissée si disponible
    #     if hasattr(rectangle, 'smoothed_angular_velocity'):
    #         print(f"smoothed_angular_velocity={rectangle.smoothed_angular_velocity:.2f} °/s")
    #     print(f"{ball.vel[0]=}, {ball.vel[1]=}, {ball.angular_speed=}")

    # Repositionnement pour éviter l'enfoncement
    ball.pos[0] = contact[0] + normal[0] * ball.radius
    ball.pos[1] = contact[1] + normal[1] * ball.radius

    if face and face.startswith('corner_'):
        ratio = corner_ratio if corner_ratio is not None else 0.0
        transfer = ratio * ball.vel[1]

        # Direction horizontale
        if 'hg' in face or 'bg' in face:  # gauche
            ball.vel[0] = -abs(ball.vel[0]) - abs(transfer)
            is_left_corner = True
        else:  # droite
            ball.vel[0] = abs(ball.vel[0]) + abs(transfer)
            is_left_corner = False

        # Direction verticale
        if 'hg' in face or 'hd' in face:  # haut
            ball.vel[1] = -abs(ball.vel[1]) * (1 - 0.5 * ratio)
        else:  # bas
            ball.vel[1] = abs(ball.vel[1]) * (1 - 0.5 * ratio)

        # Ajustement spin
        ball.angular_speed = adjust_spin_for_corner(ball.angular_speed, ratio, is_left_corner=is_left_corner)

        # Réduction vitesses
        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], est_mousse)

    else:
        # appliquer la réflexion selon la normale
        v_dot_n = ball.vel[0]*normal[0] + ball.vel[1]*normal[1]
        ball.vel[0] -= 2 * v_dot_n * normal[0]
        ball.vel[1] -= 2 * v_dot_n * normal[1]
        
        # print(f"face={face}, {normal=}, {tangent=}")
        # if est_mousse:
        #     print(f"  [1] Après réflexion: vel={ball.vel[0]:.2f}, {ball.vel[1]:.2f}")

        # === TRANSFERT VITESSE RAQUETTE → BALLE (seulement pour la mousse/raquette) ===
        if est_mousse:
            # Facteur de transfert de vitesse
            velocity_transfer = 1.2  # 120% de la vitesse transférée (amplifié pour plus de dynamique)
            
            # Ajouter la vitesse de la raquette à la balle
            ball.vel[0] += vel_x * velocity_transfer
            ball.vel[1] += vel_y * velocity_transfer
            
            # # if est_mousse:
                # print(f"  [2] Après transfert raquette: vel={ball.vel[0]:.2f}, {ball.vel[1]:.2f}")
            
            # === GÉNÉRATION DE SPIN BASÉE SUR OÙ ON TAPE LA BALLE ===
            # Vecteur du centre de la balle vers le point de contact
            contact_offset_x = contact[0] - ball_center_x
            contact_offset_y = contact[1] - ball_center_y
            
            # Normaliser par le rayon pour avoir un ratio [-1, 1]
            # contact_ratio_y > 0 = on tape en dessous de la balle (backspin/coupé)
            # contact_ratio_y < 0 = on tape au dessus de la balle (topspin)
            contact_ratio_y = contact_offset_y / ball.radius if ball.radius > 0 else 0
            
            # La vitesse horizontale de la raquette amplifie l'effet
            # Plus on frappe fort horizontalement, plus l'effet est marqué
            paddle_speed = np.sqrt(vel_x**2 + vel_y**2)
            
            # Générer le spin :
            # - Taper dessus (contact_ratio_y < 0) + mouvement vers la droite (vel_x > 0) = topspin (positif)
            # - Taper dessous (contact_ratio_y > 0) + mouvement vers la droite (vel_x > 0) = backspin (négatif)
            # Le signe du spin dépend de : -contact_ratio_y * signe(vel_x)
            spin_generation = 0.5  # facteur de conversion amplifié (était 0.2)
            direction_x = 1 if vel_x >= 0 else -1  # direction du coup
            generated_spin = -contact_ratio_y * paddle_speed * spin_generation * direction_x
            
            ball.angular_speed += generated_spin

            # === SPIN GÉNÉRÉ PAR LA VITESSE ANGULAIRE DE LA RAQUETTE ===
            # Convention demandée :
            # - vitesse angulaire > 0 => accentue spin horaire (positif)
            # - vitesse angulaire < 0 => accentue spin antihoraire (négatif)
            if hasattr(rectangle, 'smoothed_angular_velocity'):
                angular_spin_factor = 0.02  # facteur de conversion deg/s -> spin
                angular_dir = 1 if rectangle.smoothed_angular_velocity >= 0 else -1
                angular_strength = abs(rectangle.smoothed_angular_velocity)
                generated_spin_wrist = angular_strength * angular_spin_factor * angular_dir
                ball.angular_speed += generated_spin_wrist

        # Ratio basé sur le spin (existant)
        max_spin = 500.0
        ratio = min(1.0, abs(ball.angular_speed) / max_spin)

        # Redistribution de l'énergie
        ball.vel[0] += abs(ball.angular_speed) * spin_factor * (ratio if signe_x else -ratio)
        ball.vel[1] += abs(ball.angular_speed) * spin_factor * (ratio) * (1 if signe_y else -1) # si dans le même sens alors on descend (positif) sinon on monte (negatif)
        
        # if est_mousse:
        #    print(f"  [3] Après redistribution spin: vel={ball.vel[0]:.2f}, {ball.vel[1]:.2f}")

        ball.angular_speed *= 0.8

        ball.vel[0], ball.vel[1] = reduction_speed(ball.vel[0], ball.vel[1], est_mousse)
        
        # if est_mousse:
        #    print(f"  [4] Après réduction: vel={ball.vel[0]:.2f}, {ball.vel[1]:.2f}")

    # if est_mousse:
    #     print("Après collision:")
    #     print(f"{vel_x=}, {vel_y=}")
    #     print(f"{ball.vel[0]=}, {ball.vel[1]=}, {ball.angular_speed=}")





def check_table_collision(ball, table):
    """Vérifie la collision avec la table et track les rebonds."""
    from config import WIDTH
    
    # Sauvegarder la vitesse Y avant collision
    old_vel_y = ball.vel[1]
    old_pos_y = ball.pos[1]
    
    check_rect_collision(ball, table, est_mousse=False, est_table=True, a=0.22)
    
    # Si la balle allait vers le bas et maintenant va vers le haut = rebond
    if old_vel_y > 0 and ball.vel[1] < 0:
        net_center = WIDTH // 2
        if ball.pos[0] < net_center:
            ball.bounces_left += 1
        else:
            ball.bounces_right += 1
        return True
    
    return False



def check_ball_paddle(ball, paddle, screen):
    # Vérifier si la raquette peut toucher la balle
    if not paddle.can_hit:
        return
    
    # Sauvegarder la position avant collision pour détecter si collision a eu lieu
    old_pos = ball.pos.copy()
    
    check_rect_collision(ball, paddle, est_mousse=True, est_table=False, a=0.35, screen=screen)

    
    # Si la position a changé, une collision a eu lieu
    if not np.array_equal(old_pos, ball.pos):
        paddle.can_hit = False  # La raquette ne peut plus toucher la balle


def check_ball_net(ball, net, restitution=RESTITUTION, spin_factor=0.3, spin_damping=0.8):
    """
    Gestion de la collision avec le filet :
    - ball : instance de Ball
    - net : instance de Net
    - restitution : coefficient pour vel_x inversé
    - spin_factor : influence de l'angular_speed sur vel_y
    - spin_damping : perte de spin après collision
    """
    check_rect_collision(ball, net, est_mousse=False, est_table=False, a=0.22)


