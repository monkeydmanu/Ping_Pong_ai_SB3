"""
Script de test pour vérifier l'environnement et le feature extractor SB3.

Usage:
    python test_sb3_env.py
"""

import numpy as np
from ai.environment import PingPongEnv
from ai.feature_extractor import HybridFeatureExtractor
from stable_baselines3.common.env_checker import check_env
import torch as th
import gymnasium as gym


def test_env_basic():
    """Test basique de l'environnement."""
    print("="*70)
    print("🧪 Test 1: Création et reset de l'environnement")
    print("="*70)
    
    env = PingPongEnv(render_mode=None)
    
    # Reset
    obs, info = env.reset()
    
    print(f"\n✅ Reset réussi!")
    print(f"   Type observation: {type(obs)}")
    print(f"   Clés: {list(obs.keys())}")
    print(f"   Shapes:")
    for key, value in obs.items():
        print(f"     - {key}: {value.shape}, dtype={value.dtype}")
    
    # Vérifier les valeurs
    print(f"\n📊 Valeurs de l'observation:")
    print(f"   ball_idx: {obs['ball_idx']}")
    print(f"   paddle_idx: {obs['paddle_idx']}")
    print(f"   angle_idx: {obs['angle_idx']}")
    print(f"   continuous (premiers 5): {obs['continuous'][:5]}")
    
    env.close()
    return True


def test_env_step():
    """Test d'un step de l'environnement."""
    print("\n" + "="*70)
    print("🧪 Test 2: Exécution d'un step")
    print("="*70)
    
    env = PingPongEnv(render_mode=None)
    obs, _ = env.reset()
    
    # Action aléatoire
    action = env.action_space.sample()
    print(f"\n🎮 Action: {action}")
    
    # Step
    obs_next, terminated, info = env.step(action)
    
    print(f"\n✅ Step réussi!")
    print(f"   Terminated: {terminated}")
    print(f"   Info keys: {list(info.keys())}")
    print(f"   Steps: {info.get('steps', 'N/A')}")
    print(f"   Agent hits: {info.get('agent_hits', 'N/A')}")
    
    env.close()
    return True


def test_env_checker():
    """Utilise le checker officiel de SB3."""
    print("\n" + "="*70)
    print("🧪 Test 3: Vérification SB3 (check_env)")
    print("="*70)
    
    env = PingPongEnv(render_mode=None)
    
    try:
        check_env(env, warn=True)
        print("\n✅ Environnement compatible avec SB3!")
        env.close()
        return True
    except Exception as e:
        print(f"\n❌ Erreur de compatibilité: {e}")
        env.close()
        return False


def test_feature_extractor():
    """Test du feature extractor."""
    print("\n" + "="*70)
    print("🧪 Test 4: HybridFeatureExtractor")
    print("="*70)
    
    # Créer un observation_space factice
    from gymnasium import spaces
    observation_space = spaces.Dict({
        "ball_idx": spaces.Box(low=0, high=255, shape=(1,), dtype=np.int64),
        "paddle_idx": spaces.Box(low=0, high=255, shape=(1,), dtype=np.int64),
        "angle_idx": spaces.Box(low=0, high=15, shape=(1,), dtype=np.int64),
        "continuous": spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
    })
    
    # Créer le feature extractor
    embed_dim = 16
    extractor = HybridFeatureExtractor(observation_space, embed_dim=embed_dim)
    
    print(f"\n✅ Feature extractor créé!")
    print(f"   Embed dim: {embed_dim}")
    print(f"   Features dim (sortie): {extractor.features_dim}")
    print(f"   Expected: {3 * embed_dim + 14} = {3 * 16 + 14}")
    
    # Test forward pass avec un batch fictif
    batch_size = 4
    fake_obs = {
        "ball_idx": th.randint(0, 256, (batch_size, 1), dtype=th.long),
        "paddle_idx": th.randint(0, 256, (batch_size, 1), dtype=th.long),
        "angle_idx": th.randint(0, 16, (batch_size, 1), dtype=th.long),
        "continuous": th.randn(batch_size, 14, dtype=th.float32)
    }
    
    with th.no_grad():
        output = extractor(fake_obs)
    
    print(f"\n📊 Forward pass:")
    print(f"   Input batch size: {batch_size}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected: ({batch_size}, {extractor.features_dim})")
    
    # Vérifier les embeddings individuels
    print(f"\n🔍 Détails des embeddings:")
    ball_idx_sample = fake_obs["ball_idx"][0].item()
    with th.no_grad():
        ball_embed_sample = extractor.ball_embedding(fake_obs["ball_idx"][0])
    print(f"   Ball index [0]: {ball_idx_sample} → embedding shape: {ball_embed_sample.shape}")
    
    if output.shape == (batch_size, extractor.features_dim):
        print("\n✅ Feature extractor fonctionne correctement!")
        return True
    else:
        print("\n❌ Shape incorrecte!")
        return False


def test_full_episode():
    """Test d'un épisode complet."""
    print("\n" + "="*70)
    print("🧪 Test 5: Épisode complet (max 100 steps)")
    print("="*70)
    
    env = PingPongEnv(render_mode=None)
    obs, _ = env.reset()
    
    total_reward = 0
    steps = 0
    max_steps = 100
    
    while steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            print(f"\n🏁 Épisode terminé après {steps} steps")
            print(f"   Reward total: {total_reward:.2f}")
            print(f"   Winner: {info.get('winner_side', 'N/A')}")
            print(f"   Point message: {info.get('point_message', 'N/A')}")
            break
    
    if not terminated:
        print(f"\n⏸️  Épisode interrompu après {max_steps} steps (pas terminé)")
        print(f"   Reward accumulé: {total_reward:.2f}")
    
    env.close()
    print("\n✅ Test d'épisode terminé!")
    return True


def main():
    """Exécute tous les tests."""
    print("\n" + "🔬"*35)
    print("    TEST SUITE - Environnement SB3 Ping-Pong")
    print("🔬"*35 + "\n")
    
    tests = [
        ("Environnement basique", test_env_basic),
        ("Step", test_env_step),
        ("SB3 Compatibility", test_env_checker),
        ("Feature Extractor", test_feature_extractor),
        ("Épisode complet", test_full_episode),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ ERREUR dans {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Résumé
    print("\n" + "="*70)
    print("📋 RÉSUMÉ DES TESTS")
    print("="*70)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print("\n" + "="*70)
    if passed == total:
        print(f"🎉 Tous les tests réussis! ({passed}/{total})")
    else:
        print(f"⚠️  {passed}/{total} tests réussis")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
