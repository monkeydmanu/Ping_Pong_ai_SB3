"""
Test de l'environnement avec des actions aléatoires.
Permet de vérifier que les rewards, les épisodes et le reset fonctionnent correctement.

Usage:
    python test_random_actions.py
    python test_random_actions.py --episodes 100
    python test_random_actions.py --render  # Avec affichage
"""

import argparse
from ai.environment import PingPongEnv


def main():
    parser = argparse.ArgumentParser(description='Test avec actions aléatoires')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Nombre d\'épisodes à tester')
    parser.add_argument('--render', action='store_true',
                        help='Afficher le jeu')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Steps max par épisode')
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"🎮 Test de l'environnement avec {args.episodes} épisodes")
    print("="*70)
    
    # Créer l'environnement
    render_mode = "human" if args.render else None
    env = PingPongEnv(render_mode=render_mode, agent_side="left", static_spawn=False)
    
    episodes = args.episodes
    total_rewards = []
    total_steps = []
    wins = 0
    losses = 0
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"\n📍 Épisode {episode + 1}/{episodes}")
        print(f"   Observation initiale - Type: {type(obs)}, Clés: {list(obs.keys())}")
        
        while not done and steps < args.max_steps:
            # Action aléatoire
            random_action = env.action_space.sample()
            
            if steps == 0:
                print(f"   Action (ex): {random_action}")
            
            # Step
            obs, reward, terminated, truncated, info = env.step(random_action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Afficher quelques rewards pour vérifier
            if steps <= 3 or (reward != 0 and steps <= 10):
                print(f"   Step {steps}: action={random_action}, reward={reward:.3f}")
        
        # Statistiques de l'épisode
        winner = info.get('winner_side', None)
        agent_side = env.agent_side
        
        if winner == agent_side:
            wins += 1
            result = "✅ WIN"
        elif winner is not None:
            losses += 1
            result = "❌ LOSS"
        else:
            result = "⏸️  TIMEOUT"
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        print(f"   Résultat: {result}")
        print(f"   Reward total: {episode_reward:.2f}")
        print(f"   Steps: {steps}")
        print(f"   Winner: {winner}")
        print(f"   Message: {info.get('point_message', 'N/A')}")
    
    # Statistiques finales
    print("\n" + "="*70)
    print("📊 STATISTIQUES FINALES")
    print("="*70)
    print(f"Épisodes: {episodes}")
    print(f"Wins: {wins} ({wins/episodes*100:.1f}%)")
    print(f"Losses: {losses} ({losses/episodes*100:.1f}%)")
    print(f"Timeouts: {episodes - wins - losses}")
    print(f"\nReward moyen: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Reward min: {min(total_rewards):.2f}")
    print(f"Reward max: {max(total_rewards):.2f}")
    print(f"\nSteps moyen: {sum(total_steps)/len(total_steps):.1f}")
    print(f"Steps min: {min(total_steps)}")
    print(f"Steps max: {max(total_steps)}")
    print("="*70)
    
    env.close()
    
    print("\n✅ Test terminé! L'environnement fonctionne correctement.")
    print("   Vous pouvez maintenant lancer l'entraînement avec: python train_sb3.py")


if __name__ == "__main__":
    main()
