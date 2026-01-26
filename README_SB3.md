# Ping-Pong RL avec Stable-Baselines3

Entraînement d'un agent PPO pour jouer au ping-pong avec des embeddings spatiaux et Stable-Baselines3.

## 🚀 Installation

Les dépendances sont déjà dans `requirements.txt` :

```bash
pip install -r requirements.txt
```

## 📋 Fichiers importants

### Environnement et Architecture
- **`ai/environment.py`** : Environnement Gymnasium avec Dict observation space
- **`ai/feature_extractor.py`** : Feature extractor personnalisé avec embeddings spatiaux
- **`config.py`** : Configuration du jeu (dimensions, physique, etc.)

### Scripts d'entraînement
- **`train_sb3.py`** : Entraînement PPO avec SB3 (RECOMMANDÉ)
- **`train.py`** : Ancien entraînement PyTorch manuel (legacy)

### Scripts de test
- **`check_env_sb3.py`** : Vérifie la compatibilité SB3 avec `check_env()`
- **`test_random_actions.py`** : Test avec actions aléatoires
- **`test_sb3_env.py`** : Suite de tests complète

## 🧪 Vérifier l'environnement

### 1. Vérification rapide avec check_env
```bash
python check_env_sb3.py
```

### 2. Test avec actions aléatoires (50 épisodes)
```bash
python test_random_actions.py
```

Avec affichage :
```bash
python test_random_actions.py --render --episodes 10
```

### 3. Suite de tests complète
```bash
python test_sb3_env.py
```

## 🎮 Entraînement

### Entraînement basique (100k timesteps)
```bash
python train_sb3.py
```

### Entraînement personnalisé
```bash
python train_sb3.py --timesteps 500000 --learning-rate 3e-4 --n-steps 2048
```

### Continuer un entraînement existant
```bash
python train_sb3.py --load models_sb3/checkpoints/ppo_pingpong_10000_steps.zip --timesteps 100000
```

### Avec affichage (ralentit beaucoup)
```bash
python train_sb3.py --render --timesteps 10000
```

### Options complètes
```bash
python train_sb3.py \
    --timesteps 500000 \
    --learning-rate 3e-4 \
    --n-steps 2048 \
    --batch-size 64 \
    --n-epochs 10 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --ent-coef 0.01 \
    --embed-dim 16 \
    --save-freq 10000
```

## 📊 Suivi de l'entraînement

Lancer TensorBoard :
```bash
tensorboard --logdir=./logs_sb3/tensorboard/
```

Puis ouvrir http://localhost:6006

## 🏗️ Architecture

### Observation Space (Dict)
```python
{
    "ball_idx": Box(0, 255, (1,), int64),      # Position balle (grille 16x16)
    "paddle_idx": Box(0, 255, (1,), int64),    # Position raquette (grille 16x16)
    "angle_idx": Box(0, 15, (1,), int64),      # Angle raquette (16 bins)
    "continuous": Box(-1, 1, (14,), float32)   # Features continues
}
```

### Features continues (14 valeurs)
- `[0-1]` : Vitesse balle (vx, vy)
- `[2]` : Spin balle
- `[3-4]` : Vitesse raquette (vx, vy)
- `[5]` : Balle de notre côté ? (±1)
- `[6]` : Balle vient vers nous ? (±1)
- `[7]` : Rebonds notre côté (0/0.5/1)
- `[8]` : Rebonds côté adverse (0/0.5/1)
- `[9]` : Service ? (±1)
- `[10-11]` : **Distance balle→raquette (dx, dy)** ← Nouvelle feature !
- `[12-13]` : Position raquette (x, y)

### Action Space (Continu)
```python
Box(-1, 1, (3,), float32)  # [move_x, move_y, rotate]
```

### Feature Extractor (HybridFeatureExtractor)
1. **Embeddings spatiaux** (3 × 16 dim) :
   - Ball position (256 cellules → 16 dim)
   - Paddle position (256 cellules → 16 dim)
   - Paddle angle (16 bins → 16 dim)
2. **Features continues** (14 dim) directement concaténées
3. **Output** : 62 features → Réseaux Actor/Critic (2 × [256, 256])

## 🎯 Avantages de SB3

✅ **Stabilité** : Implémentation PPO testée et optimisée  
✅ **Debuggage** : TensorBoard intégré, métriques automatiques  
✅ **Callbacks** : Sauvegarde auto, évaluation périodique  
✅ **Flexibilité** : Facile de changer hyperparamètres  
✅ **Embeddings** : Feature extractor personnalisé gardé  

## 📁 Structure des sauvegardes

```
models_sb3/
├── checkpoints/           # Sauvegardes périodiques
│   ├── ppo_pingpong_10000_steps.zip
│   ├── ppo_pingpong_20000_steps.zip
│   └── ...
├── best/                  # Meilleur modèle (évaluation)
│   └── best_model.zip
└── ppo_pingpong_final.zip # Modèle final

logs_sb3/
├── tensorboard/           # Logs TensorBoard
├── eval/                  # Logs d'évaluation
└── monitor.csv            # Métriques d'entraînement
```

## 🔧 Hyperparamètres recommandés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `learning_rate` | 3e-4 | Learning rate Adam |
| `n_steps` | 2048 | Steps avant update |
| `batch_size` | 64 | Taille mini-batches |
| `n_epochs` | 10 | Epochs par update |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO clip range |
| `ent_coef` | 0.01 | Coefficient entropie |
| `embed_dim` | 16 | Dimension embeddings |

## 🐛 Troubleshooting

### Erreur : "observation space mismatch"
→ Vérifier que l'environnement retourne bien un Dict avec les bonnes clés et shapes

### Erreur : "embedding index out of range"
→ Vérifier que ball_idx, paddle_idx ∈ [0, 255] et angle_idx ∈ [0, 15]

### Performance faible
→ Augmenter `n_steps` (plus de données avant update)  
→ Réduire `learning_rate` (apprentissage plus stable)  
→ Augmenter `ent_coef` (plus d'exploration)

### NaN dans les losses
→ Vérifier que les rewards sont bien normalisés  
→ Activer gradient clipping (déjà activé par défaut)

## 📚 Ressources

- [SB3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Custom Environments](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)
- [PPO Algorithm](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
