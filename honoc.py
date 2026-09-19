import numpy as np
import plotly.graph_objects as go

def courbe_valeurs(y_obs, pred_avant_vals, pred_apres_vals, titre="Observe vs predit (Test)"):
    y_obs = np.asarray(y_obs, dtype=float)
    pred_avant_vals = np.asarray(pred_avant_vals, dtype=float)
    pred_apres_vals = np.asarray(pred_apres_vals, dtype=float)
    ordre = np.argsort(y_obs)
    x = np.arange(1, len(y_obs) + 1)

    fig = go.Figure()
    fig.add_scatter(x=x, y=y_obs[ordre], mode="lines", name="Observe",
                    line=dict(color="#141B34", width=3))
    fig.add_scatter(x=x, y=pred_avant_vals[ordre], mode="lines", name="Predit - avant tuning",
                    line=dict(color="#95A5A6", width=2, dash="dot"))
    fig.add_scatter(x=x, y=pred_apres_vals[ordre], mode="lines", name="Predit - apres tuning",
                    line=dict(color="#1F3A93", width=2, dash="dash"))
    fig.update_layout(title=titre, xaxis_title="Observations triees par valeur observee",
                      yaxis_title="Valeur", template="plotly_white", height=520,
                      legend=dict(orientation="h", y=1.08, x=.5, xanchor="center"))
    return fig

courbe_valeurs(y_te, pred_avant["Test"], pred_apres["Test"]).show()

















attendus = {k: v for k, v in params_finaux.items() if k not in ("n_jobs", "verbose")}
reels = modele_apres.get_params()
ecarts = {k: (attendus[k], reels.get(k)) for k in attendus if reels.get(k) != attendus[k]}

if ecarts:
    print("DECALAGE : modele_apres n'utilise PAS les hyperparametres tunes pour :")
    for k, (att, reel) in ecarts.items():
        print(f"  - {k} : attendu {att!r}, trouve {reel!r}")
else:
    print("OK : modele_apres utilise exactement les hyperparametres de params_finaux.")















import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-white')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = False

data_test = comparaison_test[comparaison_test["Split"] == "Test"].iloc[0]

metrics = ["MAE", "RMSE", "wMAPE"]
values_avant = [data_test[f"{m}_avant"] for m in metrics]
values_apres = [data_test[f"{m}_apres"] for m in metrics]
gains_calc = [(values_avant[i] - values_apres[i]) / abs(values_avant[i]) * 100
              for i in range(len(metrics))]

color_avant, color_apres = '#95A5A6', '#1F3A93'
color_gain_pos, color_gain_neg = '#2ECC71', '#E74C3C'

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Comparaison des Performances (Test)', fontsize=13, fontweight='bold')
x = np.arange(len(metrics))
width = 0.35

ax1 = axes[0]
bars1 = ax1.bar(x - width/2, values_avant, width, label='Avant Tuning', color=color_avant)
bars2 = ax1.bar(x + width/2, values_apres, width, label='Après Tuning', color=color_apres)
ax1.set_ylabel('Valeur')
ax1.set_title('Métriques Absolues (Test)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
for bars in (bars1, bars2):
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h:,.0f}' if h > 100 else f'{h:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8)

ax2 = axes[1]
colors_gains = [color_gain_pos if g > 0 else color_gain_neg for g in gains_calc]
bars_gain = ax2.bar(x, gains_calc, width, color=colors_gains)
ax2.set_ylabel('Gain (%)')
ax2.set_title('Amelioration Relative (%)', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.axhline(0, color='black', linewidth=0.8)
for bar, gain in zip(bars_gain, gains_calc):
    h = bar.get_height()
    ax2.annotate(f'{gain:+.2f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 2 if h > 0 else -2), textcoords="offset points",
                ha='center', va='bottom' if h > 0 else 'top', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()







