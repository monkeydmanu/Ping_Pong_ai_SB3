"""
Vérification rapide de l'environnement avec check_env de SB3.

Usage:
    python check_env_sb3.py
"""

from stable_baselines3.common.env_checker import check_env
from ai.environment import PingPongEnv


print("="*70)
print("🔍 Vérification de l'environnement Ping-Pong avec SB3")
print("="*70)

# Créer l'environnement
env = PingPongEnv(render_mode=None, agent_side="left", static_spawn=False)

print("\n📋 Observation space:", env.observation_space)
print("📋 Action space:", env.action_space)

# Vérifier avec check_env de SB3
print("\n🧪 Exécution de check_env()...")
print("-"*70)

try:
    check_env(env, warn=True)
    print("-"*70)
    print("\n✅ Environnement compatible avec Stable-Baselines3!")
    print("   Vous pouvez maintenant entraîner un agent avec PPO.")
except Exception as e:
    print("-"*70)
    print(f"\n❌ Erreur détectée: {e}")
    import traceback
    traceback.print_exc()

env.close()
print("\n" + "="*70)
