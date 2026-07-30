# Priorisation des anomalies détectées par Conformal Prediction

## Objectif

La méthode de **Split Conformal Prediction** permet de détecter les observations dont la valeur observée se situe en dehors de l'intervalle de prédiction conforme.

Une observation est considérée comme anormale lorsque

$$
y_i^{obs} \notin [L_i,U_i]
$$

où :

- $y_i^{obs}$ est la valeur observée ;
- $L_i$ est la borne inférieure de l'intervalle conforme ;
- $U_i$ est la borne supérieure.

Cependant, toutes les anomalies n'ont pas la même importance. Une légère sortie de l'intervalle sur un portefeuille de faible montant n'a pas le même impact qu'une anomalie importante sur un portefeuille représentant plusieurs centaines de millions d'euros.

L'objectif est donc de construire un **score de priorisation** permettant de classer automatiquement les anomalies de la plus critique à la moins critique.

---

# Proposition 1 : Priorisation à partir de la borne conforme

## Principe

L'idée consiste à mesurer **à quel point l'observation dépasse la limite acceptable définie par la Conformal Prediction**.

Lorsque l'observation dépasse la borne supérieure :

$$
A_i=\frac{y_i^{obs}-U_i}{U_i}
$$

Lorsque l'observation est inférieure à la borne basse :

$$
A_i=\frac{L_i-y_i^{obs}}{L_i}
$$

Une écriture unique peut être utilisée :

$$
A_i=
\frac{\min\left(|y_i^{obs}-L_i|,\;|y_i^{obs}-U_i|\right)}
{\text{borne franchie}}
$$

Ce score représente le dépassement relatif de la frontière conforme.

Plus cette valeur est grande, plus l'observation est éloignée de la zone considérée comme normale.

---

## Pourquoi diviser par la borne ?

L'utilisation de

$$
y_i^{obs}-U_i
$$

ne permet pas de comparer des portefeuilles ayant des ordres de grandeur très différents.

Par exemple :

- un dépassement de 100 € sur un portefeuille de 1 000 € représente 10 % ;
- un dépassement de 100 € sur un portefeuille de 100 M€ est quasiment négligeable.

En normalisant par la borne conforme,

$$
\frac{y_i^{obs}-U_i}{U_i},
$$

on obtient un indicateur indépendant de l'échelle des données.

Toutes les anomalies deviennent ainsi comparables entre elles.

---

# Pondération par la taille économique du portefeuille

Une anomalie statistique importante n'est pas forcément prioritaire si elle concerne un portefeuille très peu exposé.

Inversement, une anomalie relativement faible peut devenir critique lorsqu'elle concerne un portefeuille représentant une exposition financière importante.

Pour intégrer cette information métier, le score précédent est pondéré par la **Gross Written Premium (GWP)**.

Le score devient

$$
Score_i=A_i\times GWP_i
$$

c'est-à-dire

$$
Score_i=
\left(
\frac{|y_i^{obs}-\text{borne}|}
{\text{borne}}
\right)
\times GWP_i.
$$

Les anomalies sont ensuite classées par ordre décroissant de ce score.

Cette approche permet de privilégier les anomalies ayant simultanément :

- un fort dépassement de la borne conforme ;
- un impact économique important.

---

# Proposition 2 : Priorisation à partir de l'erreur du modèle

## Principe

Une seconde possibilité consiste à utiliser directement la prédiction du modèle au lieu de la borne conforme.

On calcule alors l'erreur relative :

$$
B_i=
\frac{|y_i^{obs}-\hat y_i|}
{\hat y_i}
$$

où

- $\hat y_i$ désigne la prédiction du modèle.

Cette quantité mesure l'écart relatif entre la valeur réellement observée et la valeur attendue par le modèle.

Plus cette erreur est importante, plus le comportement observé est inhabituel.

---

# Pondération par la GWP

Comme précédemment, on intègre l'importance économique du portefeuille.

Le score devient

$$
Score_i=
\frac{|y_i^{obs}-\hat y_i|}
{\hat y_i}
\times GWP_i.
$$

Les observations sont ensuite triées par ordre décroissant.

---

# Différence entre les deux approches

La première approche mesure :

> **À quel point l'observation dépasse la limite définie par la Conformal Prediction.**

La seconde mesure :

> **À quel point le modèle s'est trompé sur cette observation.**

Ces deux informations sont complémentaires.

Une observation peut être :

- très éloignée de la prédiction du modèle,
- tout en restant à l'intérieur de l'intervalle conforme.

À l'inverse,

une observation peut sortir légèrement de l'intervalle conforme tout en restant relativement proche de la prédiction.

Les deux approches permettent donc de capturer des aspects différents de l'anomalie.

---

# Proposition d'amélioration

Une approche plus complète consiste à combiner simultanément :

- la gravité statistique de l'anomalie ;
- la qualité de la prédiction du modèle ;
- l'importance économique du portefeuille.

On peut définir le score suivant :

$$
Score_i=
\left(
\frac{|y_i^{obs}-\hat y_i|}
{\hat y_i}
\right)
\times
\left(
\frac{|y_i^{obs}-\text{borne}|}
{\text{borne}}
\right)
\times
GWP_i.
$$

Une anomalie sera alors considérée comme prioritaire lorsqu'elle vérifie simultanément les trois propriétés suivantes :

- elle est très éloignée de la prédiction du modèle ;
- elle dépasse largement l'intervalle conforme ;
- elle concerne un portefeuille présentant une exposition financière importante.

Cette approche fournit un classement des anomalies beaucoup plus proche des besoins opérationnels des équipes actuarielles.

---

# Perspectives d'amélioration

L'utilisation directe de la **Gross Written Premium (GWP)** peut conduire à favoriser systématiquement les plus grands portefeuilles.

Une amélioration possible consiste à remplacer la GWP par une version normalisée.

Par exemple,

$$
GWP_i^{*}=\log(1+GWP_i)
$$

ou

$$
GWP_i^{*}=\frac{GWP_i}{\max(GWP)}.
$$

Le score devient alors

$$
Score_i=
\left(
\frac{|y_i^{obs}-\text{borne}|}
{\text{borne}}
\right)
\times
\log(1+GWP_i)
$$

ou

$$
Score_i=
\left(
\frac{|y_i^{obs}-\hat y_i|}
{\hat y_i}
\right)
\times
\left(
\frac{|y_i^{obs}-\text{borne}|}
{\text{borne}}
\right)
\times
\log(1+GWP_i).
$$

Cette normalisation permet d'éviter qu'un très grand portefeuille domine entièrement le classement tout en conservant la notion d'impact économique.
