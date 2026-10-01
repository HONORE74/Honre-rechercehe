## Objet

Cette issue présente la démarche de détection des observations atypiques sur l'indicateur `Claims_incurred`, en deux temps :

1. **pourquoi nous considérons certaines observations comme des anomalies** ;
2. **comment lire le tableau de bord** qui présente ces anomalies pour le Q4 2024.

> ℹ️ Les chiffres présentés (135 sous-portefeuilles, 16 anomalies, scores, montants) sont ceux du tableau de bord construit sur les données de démonstration.

```mermaid
flowchart LR
    P["Prédiction"] --> I["Intervalle conforme à 90 %"]
    I --> Q{"Observé dans l'intervalle ?"}
    Q -- "Oui" --> N["Variation normale"]
    Q -- "Non" --> S["Anomalie potentielle"]
    S --> SC["Score de priorisation A × B × GWP"]
    SC --> PR["Classement du plus critique au moins critique"]
    PR --> INV["Investigation : corriger ou documenter"]
```

---

# Partie 1 — Pourquoi considérons-nous certaines observations comme des anomalies ?

## 1.1 Définition

Une anomalie est une observation dont la valeur est **incompatible avec ce que son contexte permettait d'attendre**, compte tenu des primes, de l'historique et de la saisonnalité.

## 1.2 Le modèle prédit une valeur attendue, pas la valeur exacte

Le modèle que nous avons entraîné prédit la **valeur centrale attendue** et non la valeur exacte, parce que les sinistres comportent naturellement une part d'aléa. Le modèle prédit donc une **valeur attendue**, et pas une valeur qui devrait être exactement égale à l'observé.

## 1.3 L'apport de la Conformal Prediction

C'est dans ce cas que la Conformal Prediction devient intéressante. Elle nous permet de ne plus nous poser la question « l'observé est-il exactement égal à la prédiction ? », mais plutôt :

> **L'observé est-il compatible avec ce que le modèle considère comme une variation normale ?**

C'est ce qui nous amène à construire la **zone de prédiction** :

```math
[ borne basse ; borne haute ]
```

**Par exemple**, avec une prédiction $`\hat{y} = 100`$ et un intervalle $`[80\ ;\ 120]`$ : **95 est normal**, mais **125 ne l'est pas**, car 125 se situe au-delà de la variabilité normale.

## 1.4 Historique de la prédiction : le modèle a fait ses preuves

Sur l'historique, chaque prédiction est faite **avant** de connaître l'observé, et l'observé tombe à chaque fois dans son intervalle. Depuis l'entraînement, le modèle a donc fait ses preuves sur ce sous-portefeuille.

En 2024-Q4, l'observé sort nettement de la borne. Ce n'est donc pas le modèle qui décroche : c'est plutôt **la donnée qui rompt avec un comportement bien prédit jusque-là**. Le modèle s'appuie par ailleurs sur les variables explicatives importantes pour faire sa prédiction.

> 🖼️ **Image à insérer ici :** l'historique du sous-portefeuille Delta Conso | FR-NonVie | Creditor | Unemployment avec ses prédictions successives et leurs intervalles (backtest). L'observé reste dans la bande bleue trimestre après trimestre, puis en sort nettement en 2024-Q4.

## 1.5 Sortir de l'intervalle : un premier niveau de détection

Cependant, sortir de l'intervalle ne veut pas dire automatiquement que l'on est en présence d'une anomalie, car l'intervalle est calibré à **90 % de couverture**. On s'attend donc à ce que 90 % de nos observations soient dans l'intervalle, et à ce que **10 % ne soient pas couvertes**.

Même si aucune donnée n'était erronée, nous aurions donc toujours environ 10 % d'observations non couvertes : sur 135 sous-portefeuilles, environ $`0{,}10 \times 135 \approx 13{,}5`$ sorties sont attendues par le seul hasard.

La sortie de l'intervalle constitue donc un **premier niveau de détection** : elle désigne des **anomalies potentielles**.

> 🖼️ **Image à insérer ici :** le nombre de sorties attendues par le seul hasard avec une couverture de 90 %, comparé au nombre de sorties observées (panneau 1 de la figure de justification).

## 1.6 Le score de priorisation

Par la suite, nous avons utilisé le **score de priorisation** pour distinguer les sorties compatibles avec l'aléa des écarts réellement préoccupants. Le score cherche donc à répondre aux deux questions suivantes.

**① De combien l'observation sort-elle de l'intervalle ?**

```math
A=∣ yobs−borne franchie ∣largeur de l’intervalle
```

**② De combien l'observation s'éloigne-t-elle de la prédiction du modèle ?**

```math
B=∣ yobs−y^ ∣∣ y^ ∣
```

Ensuite, le **GWP** (`avg_dec_Claims_incurred`) est pris en compte pour intégrer l'enjeu économique du sous-portefeuille.

### Exemple pratique pour A

Prenons $`\hat{y} = 100`$ et l'intervalle $`[80\ ;\ 120]`$, de largeur $`120 - 80 = 40`$.

- **Cas 1 :** $`y_{\text{obs}} = 121`$. L'observation dépasse la borne de seulement 1.
  ```math
  A=121−12040=0,025
  ```
  L'observation sort très peu de l'intervalle.

- **Cas 2 :** $`y_{\text{obs}} = 150`$.
  ```math
  A=150−12040=0,75
  ```
  L'observation est beaucoup plus loin de la frontière de l'intervalle.

$`A`$ répond donc à la question : **à quel point sommes-nous sortis de la zone de variabilité ?**

### Exemple pratique pour B

Supposons $`\hat{y} = 100`$.

- **Observation 1 :** $`y_{\text{obs}} = 105`$, soit $`B = \dfrac{\lvert 105 - 100 \rvert}{100} = 5\,\%`$. L'écart n'est pas énorme.
- **Observation 2 :** $`y_{\text{obs}} = 160`$, soit $`B = \dfrac{\lvert 160 - 100 \rvert}{100} = 60\,\%`$. L'écart est beaucoup plus important.

$`B`$ répond donc à la question : **l'observation est-elle éloignée de ce que le modèle prévoyait ?**

### Pourquoi ce n'est ni une erreur du modèle, ni une erreur de l'intervalle, ni une évolution normale

Le `score_composite` combine trois composantes, chacune écartant une explication concurrente :

```math
score_composite=norm⁡(A×B)×GWP,norm⁡(A×B)∈[0 ; 1]
```

| **Composante** | **Ce qu'elle mesure** | **Explication qu'elle écarte** |
| :------------- | :-------------------- | :----------------------------- |
| **A** : sortie de l'intervalle | De combien l'observé dépasse la borne franchie (basse ou haute), rapporté à la largeur de l'intervalle | **L'imprécision de l'intervalle** : une sortie de quelques pourcents de la largeur relève du hasard statistique (95 % des sorties dues au hasard ont A ≤ 0,42) |
| **B** : erreur relative | L'écart à la prédiction, rapporté à la prédiction | **L'erreur ordinaire du modèle** : l'erreur médiane est de 9 % et 95 % des erreurs normales sont sous 26 % |
| **GWP** (`avg_dec`) | Le niveau habituel du sous-portefeuille | **L'effet de taille** : à gravité égale, l'écart pèse plus sur un portefeuille important |

Le produit $`A \times B`$ n'est élevé que si l'observation est **à la fois** loin de sa borne **et** loin de la prédiction. Sur la figure ci-dessous, cela correspond au quadrant rouge : il contient les 6 vraies anomalies et aucune sortie due au hasard.

L'**évolution normale** du portefeuille est déjà prise en compte en amont : le modèle utilise les primes et l'historique, si bien qu'une hausse d'activité (primes +70 %, sinistres +88 %) reste dans son intervalle.

Enfin, le classement se lit **du plus critique au moins critique**. Les sorties dues au hasard, qui restent possibles, arrivent en fin de liste : ici, les 5 premiers rangs sont tous de vraies anomalies.

> 🖼️ **Image à insérer ici :** la carte du score. Chaque sortie y est placée selon A et B, la taille des bulles correspond au GWP, et le quadrant rouge contient les vraies anomalies (panneau 2 de la figure de justification).

## 1.7 Sens de l'écart et conséquence financière (`Claims_incurred`)

| **Situation** | **Lecture** | **Risque si la donnée est erronée** |
| :------------ | :--------- | :---------------------------------- |
| Observé **sous la borne basse** (prédiction au-dessus de l'observé) | Les sinistres enregistrés sont **sous-estimés** : flux manquant, décalage de période, erreur de signe | Provisions insuffisantes (*best estimate* Solvabilité II, passif pour sinistres survenus IFRS 17) : **perte financière** à venir et risque prudentiel |
| Observé **au-dessus de la borne haute** (prédiction en dessous de l'observé) | Les sinistres enregistrés sont **surestimés** : doublon, erreur d'unité, mauvais rattachement | Provisions et capital **mobilisés inutilement**, résultat dégradé à tort |

Si la vérification confirme un événement réel (par exemple un sinistre important), la donnée est juste : l'anomalie est alors **documentée**, sans être corrigée.

## 1.8 En résumé

> Une observation est signalée parce qu'elle **sort de l'intervalle conforme**. Comme une couverture de 90 % laisse attendre environ 10 % de sorties, ce n'est qu'une **anomalie potentielle**. Le **score de priorisation** la confirme et la hiérarchise : A écarte l'imprécision de l'intervalle, B l'erreur normale du modèle, et le GWP pondère par l'enjeu. On examine les anomalies **du plus critique au moins critique**, en sachant que le sens de l'écart indique le risque financier : sous-estimation (perte) ou surestimation (ressources immobilisées).

---

# Partie 2 — Lecture du tableau de bord (Q4 2024)

## 2.1 Présentation du tableau de bord

> 🖼️ **Image à insérer ici :** l'en-tête du tableau de bord (titre `Claims_incurred`, période validée Q4 2024, couverture cible 90 %, 135 sous-portefeuilles, 16 anomalies).

Le tableau de bord porte sur l'indicateur `Claims_incurred` et présente les résultats de la détection des observations atypiques pour le quatrième trimestre 2024 (Q4 2024). Cette période correspond au trimestre retenu pour la validation des prédictions issues du modèle. Les différents éléments du tableau de bord sont ainsi centrés sur cette période, à l'exception des représentations consacrées à l'évolution historique de l'indicateur.

Les intervalles de prédiction sont construits avec une couverture cible de 90 %, correspondant à un niveau de confiance de $`1 - \alpha`$ avec $`\alpha = 0{,}10`$. Cette couverture constitue le niveau de référence utilisé pour apprécier les observations situées en dehors des intervalles attendus et, par conséquent, susceptibles de présenter un comportement atypique.

Pour le Q4 2024, le tableau de bord porte sur **135 sous-portefeuilles** prédits. Parmi les observations analysées, **16 anomalies** sont identifiées. Ces anomalies correspondent aux lignes pour lesquelles le `score_composite` est strictement positif, ce score étant utilisé pour caractériser le degré d'atypicité des observations selon les critères retenus dans l'approche de détection.

Ainsi, le tableau de bord permet de passer d'une vision globale des prédictions à une identification ciblée des observations nécessitant une attention particulière. La présence de 16 anomalies parmi les 135 sous-portefeuilles analysés fournit un premier aperçu de l'étendue des observations atypiques détectées sur le périmètre du Q4 2024.

## 2.2 Interprétation des cartes indicateurs

> 🖼️ **Image à insérer ici :** les six cartes indicateurs (Anomalies, Number of lines, Période concernée, Moyenne, Rank, Coverage).

Les cartes indicateurs fournissent une synthèse des principaux résultats obtenus sur le périmètre étudié pour le Q4 2024. L'analyse porte sur 135 sous-portefeuilles, correspondant à un total de **2 700 lignes** dans `df_model`, soit 135 sous-portefeuilles observés sur 20 trimestres.

Sur ce périmètre, **16 anomalies** sont détectées, soit **11,9 %** des 135 sous-portefeuilles analysés. Ce résultat indique qu'une proportion limitée du périmètre présente un comportement identifié comme atypique selon le score de détection retenu. La valeur affichée ne correspond donc pas à la totalité des observations de `df_model`, mais au nombre de sous-portefeuilles classés comme anomalies pour la période considérée.

La période d'analyse correspond au Q4 2024, qui constitue la période validée par le modèle. Les statistiques associées à la variable étudiée montrent une **moyenne de 973 044**, avec des valeurs comprises entre **−732 296** et **5 452 841**. L'écart important entre le minimum et le maximum traduit une dispersion marquée des valeurs observées sur le périmètre. La présence d'une valeur minimale négative est également visible dans les résultats et correspond à l'erreur de signe classée au rang 2 (Beta Retail | FR-Vie).

Le classement permet ensuite d'identifier l'observation présentant le niveau d'anomalie le plus élevé sur le périmètre. Le **rang n°1** est associé au sous-portefeuille « Gamma Credit | IT-Vie | Protection | Disability ». Cette information permet de cibler les observations qui nécessitent une attention particulière dans le cadre de l'analyse des anomalies.

Enfin, la **couverture observée est de 87,4 %**, soit 118 sous-portefeuilles sur 135 dont la valeur observée se situe à l'intérieur de l'intervalle de prédiction. Cette valeur est inférieure à la couverture cible de 90 % définie pour le modèle. Il existe donc **17 sous-portefeuilles situés en dehors de l'intervalle de prédiction**. La couverture constitue ainsi un indicateur important pour apprécier l'adéquation entre les intervalles produits par le modèle et les observations effectivement réalisées.

Cette couverture inférieure à la cible signifie que les observations sortant des intervalles sont plus nombreuses que ce qui serait attendu avec une couverture de 90 % : 17 sorties, contre environ 13,5 attendues. Elle constitue donc un signal à examiner dans l'analyse de la qualité des intervalles et de la présence éventuelle de comportements atypiques. Toutefois, la couverture seule ne permet pas d'attribuer automatiquement ces écarts à des anomalies réelles : cette identification repose sur les critères complémentaires utilisés dans la procédure de détection, notamment le `score_composite`.

> **Pourquoi 17 sorties mais 16 anomalies ?** Le score est normalisé entre 0 et 1 sur l'ensemble des sorties : la moins grave d'entre elles reçoit donc un score nul. Elle n'est pas comptée parmi les anomalies, qui correspondent aux scores strictement positifs.

## 2.3 État des sous-portefeuilles

> 🖼️ **Image à insérer ici :** la barre « État des sous-portefeuilles » (une barre par sous-portefeuille, verte ou rouge).

La figure présente l'état des 135 sous-portefeuilles analysés pour le Q4 2024. Chaque barre représente un sous-portefeuille, et les observations sont ordonnées selon la hiérarchie **Partner > Companies > Lob > Risk**. Cette organisation permet de conserver la structure du portefeuille tout en facilitant la localisation des observations atypiques.

La distinction entre les deux couleurs permet d'identifier rapidement les sous-portefeuilles concernés par une anomalie. Les **barres vertes** correspondent aux sous-portefeuilles pour lesquels aucune anomalie n'a été détectée, tandis que les **barres rouges** correspondent à ceux dont le `score_composite` est strictement positif. Sur l'ensemble du périmètre, 16 anomalies sont ainsi identifiées parmi les 135 sous-portefeuilles, soit 11,9 % du périmètre.

La disposition des barres rouges permet également d'étudier la répartition des anomalies au sein du portefeuille. Certaines apparaissent de manière isolée, alors que d'autres sont relativement proches les unes des autres. Cette proximité peut constituer un premier indice permettant de rechercher des caractéristiques communes entre les sous-portefeuilles concernés, notamment au niveau du partenaire ou de la compagnie. Elle ne permet toutefois pas, à elle seule, d'établir l'existence d'une cause commune : une analyse complémentaire des caractéristiques des observations concernées reste nécessaire pour confirmer cette éventuelle relation.

Cette représentation apporte ainsi une information complémentaire au nombre global d'anomalies présenté précédemment. Alors que la carte indicateur permet de quantifier les anomalies détectées, cette visualisation permet de les situer dans la structure du portefeuille. Elle peut donc servir de point de départ à une analyse plus ciblée des sous-portefeuilles signalés, en particulier lorsque plusieurs anomalies semblent se concentrer dans une même zone de la structure hiérarchique.

Enfin, lorsque le nombre de sous-portefeuilles dépasse 160, les barres sont regroupées par tranches afin de préserver la lisibilité de la représentation. Cette modification concerne uniquement le mode d'affichage et n'affecte pas le principe de détection des anomalies.

## 2.4 Concentration des anomalies par secteur

> 🖼️ **Image à insérer ici :** le graphique circulaire avec les cases Partner et Companies cochées.

Le graphique circulaire permet d'analyser la répartition des anomalies selon les niveaux **Partner** et **Companies**. Les secteurs représentés correspondent aux regroupements définis par la structure du portefeuille. La taille de chaque secteur est proportionnelle à sa contribution à la somme des `score_composite` observés sur l'ensemble du périmètre.

Le résultat met en évidence une forte concentration du score d'anomalie sur le secteur **Gamma Credit**, qui représente à lui seul **87 %** de la somme des `score_composite` du périmètre. Cette proportion élevée s'explique notamment par la présence, dans ce secteur, de l'anomalie présentant le score le plus important : le tableau de bord indique un score de **472 199,72** pour la première anomalie (rang #1).

Cette représentation doit toutefois être interprétée avec précaution. Le fait que Gamma Credit représente 87 % du score total ne signifie pas que 87 % des anomalies appartiennent à ce secteur. Il s'agit d'une mesure de la contribution au score d'anomalie total : les anomalies rattachées à Gamma Credit concentrent une part très importante de l'intensité mesurée par le `score_composite`.

La couleur des secteurs apporte une information complémentaire. Elle correspond au `score_composite` de la première anomalie du secteur, avec une intensité croissante allant du bleu vers le rouge. Une couleur plus proche du rouge traduit ainsi la présence d'une anomalie ayant un score élevé au sein du secteur considéré. Dans le cas de Gamma Credit, la forte intensité du secteur est cohérente avec la présence de l'anomalie classée n°1.

Cette visualisation complète l'analyse précédente. Alors que la répartition des sous-portefeuilles permettait d'identifier les emplacements des anomalies, le graphique circulaire met en évidence leur concentration en termes d'intensité du score. Il permet ainsi de cibler en priorité les secteurs dans lesquels les anomalies contribuent le plus fortement au score global, puis d'approfondir l'analyse des sous-portefeuilles concernés.

## 2.5 Analyse du cercle à la maille la plus fine

> 🖼️ **Image à insérer ici :** le graphique circulaire avec les quatre cases cochées (Partner, Companies, Lob, Risk).

Cette représentation propose une lecture hiérarchique des anomalies en descendant progressivement dans les différentes dimensions du portefeuille : **Partner, Companies, Lob et Risk**. Chaque niveau du cercle correspond à une maille d'analyse plus fine, et l'anneau extérieur permet finalement d'identifier les anomalies individuellement, au niveau du sous-portefeuille.

L'intérêt principal de cette représentation est de localiser précisément une anomalie au sein de la structure du portefeuille. En partant d'un partenaire, l'analyse peut être affinée successivement jusqu'à la compagnie, puis au segment d'activité (Lob) et enfin au risque (Risk). Cette décomposition permet de passer d'une vision globale de la concentration des anomalies à l'identification du sous-portefeuille auquel elles sont rattachées.

La figure met notamment en évidence la concentration observée autour de Gamma Credit, puis permet de poursuivre l'analyse à des niveaux plus détaillés. À la maille la plus fine, chaque secteur extérieur correspond à une anomalie individuelle, et sa couleur, du bleu vers le rouge, traduit la valeur de son `score_composite`.

Cette représentation est particulièrement utile pour la phase d'investigation. Lorsqu'un secteur présentant une anomalie est identifié, il est possible de remonter dans la hiérarchie afin de déterminer précisément le partenaire, la compagnie, le segment d'activité et le risque concernés. L'utilisateur peut ensuite cliquer sur le secteur correspondant afin de filtrer le reste du tableau de bord sur ce périmètre.

La sélection des différentes dimensions permet ainsi d'adapter le niveau de granularité de l'analyse. Une lecture limitée aux premières dimensions fournit une vision plus agrégée de la concentration des anomalies, tandis que l'activation des quatre dimensions permet d'atteindre la maille du sous-portefeuille. Les données associées à chaque anomalie restent inchangées : seule la manière de les regrouper et de les visualiser évolue.

## 2.6 Identification des observations atypiques

> 🖼️ **Image à insérer ici :** le graphique en barres « Identification des observations atypiques », avec l'infobulle affichée au survol d'une barre.

Le graphique présente les **12 premières observations atypiques**, classées selon leur niveau d'anomalie, de la plus élevée à la moins élevée. Chaque barre correspond à une observation identifiée par le processus de détection et rattachée à un sous-portefeuille donné.

La **longueur de la barre** représente le montant observé de `Claims_incurred` pour l'observation considérée. Elle permet de comparer directement les montants observés entre les différentes anomalies. La majorité des valeurs représentées sont positives, mais une observation se distingue par une valeur négative : **Beta Retail | FR-Vie**, classée en deuxième position. Sa barre se situe à gauche de zéro, ce qui traduit un montant négatif pour l'indicateur étudié.

La **couleur des barres** apporte une information différente. Elle correspond au `score_composite`, avec une intensité plus importante lorsque le niveau d'anomalie est élevé. L'observation classée n°1, **Gamma Credit | IT-Vie**, apparaît ainsi en rouge, ce qui traduit le score d'anomalie le plus élevé du périmètre présenté : son `score_composite` est de **472 199,72**.

Cette distinction entre la longueur et la couleur est importante pour l'interprétation du graphique. Une valeur élevée de `Claims_incurred` ne signifie pas nécessairement qu'une observation constitue l'anomalie la plus importante. Le classement dépend du `score_composite`, qui tient compte de l'écart entre l'observation et ce qui était attendu selon le modèle et son intervalle de prédiction. Une observation peut donc présenter un montant élevé tout en ayant un score d'anomalie inférieur à celui d'une autre observation.

Le graphique permet également d'observer qu'un même libellé peut apparaître plusieurs fois. C'est le cas de **Gamma Credit | IT-Vie**, présent aux rangs **#1 et #7**. Ces deux occurrences correspondent à deux observations distinctes, rattachées au même groupe. Il ne faut donc pas interpréter la répétition d'un libellé comme une seule et même anomalie.

Enfin, le survol d'une barre permet d'obtenir des informations complémentaires sur l'observation sélectionnée : sa valeur observée, sa valeur prédite, son intervalle de prédiction et son `score_composite`. Le graphique constitue ainsi un point d'entrée pour examiner individuellement les observations signalées par le modèle.

## 2.7 Interprétation des intervalles de prédiction conformes

> 🖼️ **Image à insérer ici :** le graphique « Conformal Prediction Intervals: Observed vs. Predicted Values » (forest plot).

Le forest plot présente les **12 premières observations atypiques**, dans le même ordre que le graphique précédent. Pour chaque sous-portefeuille, trois éléments principaux sont représentés :

- la **bande bleue**, qui correspond à l'intervalle de prédiction conforme, pour une couverture cible de 90 % ;
- le **losange**, qui représente la valeur prédite par le modèle ;
- le **point rouge**, qui correspond à la valeur effectivement observée de `Claims_incurred`.

La comparaison entre la valeur observée et l'intervalle de prédiction permet d'identifier visuellement les observations qui s'écartent de la plage attendue. Lorsqu'un point rouge se situe en dehors de la bande bleue, la valeur effectivement observée n'est pas contenue dans l'intervalle de prédiction associé. Ces observations constituent des cas particulièrement intéressants pour la procédure de détection des comportements atypiques.

La **ligne pointillée** relie la borne franchie de l'intervalle à la valeur observée : elle mesure directement le **dépassement**. Plus ce dépassement est long relativement à la largeur de la bande, plus l'observation apparaît éloignée de ce qui était attendu par le modèle. Cette représentation permet donc de visualiser simultanément la prédiction, l'incertitude associée à cette prédiction et la réalisation effectivement observée.

L'observation **Gamma Credit | IT-Vie**, classée première dans les graphiques précédents, illustre particulièrement bien cette situation. La valeur observée est nettement supérieure à l'intervalle de prédiction, ce qui traduit un écart important entre le montant effectivement réalisé et la plage de valeurs attendue par le modèle. Cette observation est également associée au `score_composite` le plus élevé (472 199,72), ce qui explique son classement en première position.

À l'inverse, certaines observations présentent des écarts moins importants. C'est notamment le cas d'**Epsilon Auto | IT-Vie**, classée **#8**. La valeur observée se situe légèrement au-delà de la bande de prédiction, alors même que le montant de `Claims_incurred` est relativement élevé. Cette situation illustre un point important : l'importance du montant observé ne suffit pas à déterminer le niveau d'anomalie. Une valeur élevée peut rester relativement proche de l'intervalle attendu, compte tenu du niveau d'incertitude du modèle.

Le graphique permet ainsi de comprendre visuellement la différence entre une **valeur élevée** et une **valeur atypique**. Une observation n'est pas considérée comme atypique uniquement parce que son montant est important : c'est son positionnement par rapport à ce que le modèle prévoit, compte tenu de l'intervalle de prédiction, qui est déterminant.

## 2.8 Évolution historique et identification de l'anomalie

> 🖼️ **Image à insérer ici :** le graphique « Historique de l'observation » pour le groupe Delta Conso | FR-NonVie.

Ce graphique présente l'évolution de l'indicateur `Claims_incurred` pour le groupe **Delta Conso | FR-NonVie**, sur les dix derniers trimestres disponibles, de 2022-Q3 à 2024-Q4. La courbe additionne les montants de tous les sous-portefeuilles de ce groupe. Elle permet d'observer le comportement historique du groupe avant et pendant la période de validation.

Sur la majeure partie de la période, les valeurs évoluent dans une plage relativement proche, avec quelques fluctuations entre les trimestres. On observe notamment une hausse en 2023-Q4, suivie d'une diminution en 2024-Q1 et 2024-Q2, puis d'une nouvelle progression en 2024-Q3. Le dernier trimestre, 2024-Q4, se distingue cependant par une augmentation importante de `Claims_incurred`.

La valeur observée pour le groupe au 2024-Q4 est de **15 972 264**. Elle constitue le point le plus élevé de la série et se distingue nettement des niveaux observés au cours des trimestres précédents. Cette évolution constitue donc un premier signal d'écart par rapport au comportement historique du groupe.

> ⚠️ **Attention à la lecture :** le montant affiché entre parenthèses dans la liste de sélection (5 258 053) est celui du **sous-portefeuille** en anomalie, alors que le dernier point de la courbe (15 972 264) est le **total du groupe**. Les deux valeurs sont justes, mais ne mesurent pas la même chose.

Il est toutefois important de distinguer l'augmentation historique de la détection d'une anomalie. Le fait que la dernière valeur soit particulièrement élevée ne suffit pas, à lui seul, à conclure qu'il s'agit d'une anomalie. Pour déterminer si cette observation est effectivement atypique, il faut la comparer à la valeur attendue par le modèle ainsi qu'à l'incertitude associée à cette prédiction. C'est précisément l'objectif du graphique suivant.

## 2.9 Analyse à la maille la plus fine : prédiction et intervalle

> 🖼️ **Image à insérer ici :** le graphique « Anomalie à la maille la plus fine » pour le sous-portefeuille Delta Conso | FR-NonVie | Creditor | Unemployment.

Ce graphique affine l'analyse en considérant le sous-portefeuille **Delta Conso | FR-NonVie | Creditor | Unemployment**. Il reprend l'évolution de `Claims_incurred` sur la période historique et ajoute, pour le Q4 2024, les éléments issus du modèle de prédiction.

Pour les trimestres historiques, la courbe représente les valeurs effectivement observées. Au Q4 2024, plusieurs informations sont disponibles simultanément : le **losange** représente la valeur prédite par le modèle, la **bande** correspond à l'intervalle de prédiction conforme à 90 %, et le **point rouge** représente la valeur effectivement observée.

La comparaison entre ces éléments met en évidence un écart important. La valeur observée au Q4 2024, de **5 258 053**, se situe nettement au-dessus de la valeur prédite et en dehors de l'intervalle de prédiction conforme. L'observation réalisée est donc très éloignée de la plage de valeurs attendue par le modèle pour ce sous-portefeuille.

Cette représentation permet de comprendre pourquoi cette observation attire l'attention du dispositif de détection. L'anomalie ne résulte pas uniquement du fait que `Claims_incurred` atteint un niveau élevé : elle résulte surtout du fait que le niveau effectivement observé est nettement supérieur à ce que le modèle avait anticipé, au regard de l'intervalle d'incertitude associé à la prédiction.

L'intérêt de cette analyse à la maille la plus fine est également de pouvoir rattacher précisément l'écart observé à ses différentes dimensions : Partner, Companies, Lob et Risk. Dans le cas présenté, l'analyse conduit jusqu'au sous-portefeuille Delta Conso | FR-NonVie | Creditor | Unemployment, ce qui facilite l'identification du périmètre concerné et les éventuelles investigations complémentaires.

## 2.10 Les trois critères de tri à la maille la plus fine

> 🖼️ **Image à insérer ici :** la liste de sélection du graphique à la maille la plus fine, ouverte, avec le menu « Trier par ».

À la maille la plus fine, le tableau de bord permet de sélectionner un sous-portefeuille parmi les observations identifiées. La liste regroupe les sous-portefeuilles concernés et, par défaut, ceux-ci sont classés selon le `score_composite`. Le graphique associé se met alors à jour en fonction de l'observation sélectionnée, sans modifier les autres éléments du tableau de bord.

Trois modes de classement sont proposés : le `score_composite`, l'écart à l'intervalle et le montant observé. Chacun permet d'aborder les anomalies sous un angle différent.

### 1. Le classement par `score_composite`

Le premier critère est le `score_composite`, qui constitue le classement privilégié du tableau de bord. Il permet de hiérarchiser les anomalies en combinant la gravité relative de l'écart et l'enjeu associé à l'observation.

Ce classement évite de considérer uniquement l'ampleur brute de l'écart. Une anomalie présentant un écart important relativement à son propre niveau d'incertitude peut être davantage mise en évidence qu'une observation présentant un montant élevé, mais dont le comportement reste relativement cohérent avec l'intervalle attendu.

Dans la liste affichée, **Gamma Credit | IT-Vie | Protection | Disability**, avec un montant observé de **5 346 514**, apparaît ainsi en première position lorsque le tri est effectué selon le `score_composite`.

### 2. Le classement par écart à l'intervalle

Le deuxième critère repose sur l'écart à l'intervalle de prédiction. Il mesure dans quelle proportion l'observation se situe au-delà de l'intervalle attendu, relativement à la largeur de cet intervalle :

```math
eˊcart aˋ l’intervalle=deˊpassement de la borne largeur de l’intervalle(c’est le terme A du score)
```

Ce critère met en évidence les dépassements les plus nets, indépendamment de l'enjeu associé au montant observé. Il peut notamment faire ressortir des anomalies sur des sous-portefeuilles de faible montant, dès lors que la valeur observée s'éloigne fortement de la plage attendue.

Ce classement apporte ainsi une lecture complémentaire au `score_composite` : il répond davantage à la question de savoir à quel point l'observation s'écarte de son intervalle, plutôt qu'à celle de savoir quelle anomalie représente l'enjeu global le plus important.

### 3. Le classement par montant observé

Enfin, le tableau de bord permet de classer les observations selon leur montant observé de `Claims_incurred`. Ce mode de tri met en avant les observations présentant les montants les plus élevés et apporte donc une lecture davantage orientée vers la **matérialité**.

Cependant, un montant élevé ne signifie pas nécessairement qu'il s'agit de l'observation la plus atypique. Une valeur importante peut être cohérente avec le comportement historique du sous-portefeuille et avec l'intervalle de prédiction qui lui est associé. À l'inverse, une valeur plus faible peut constituer une anomalie importante si elle s'écarte fortement de ce qui était attendu.

## 2.11 Analyse des variables explicatives

> 🖼️ **Image à insérer ici :** le graphique « Les variables explicatives qui expliquent ce comportement » pour le groupe Gamma Credit | IT-Vie.

Cette figure présente les cinq variables numériques de `df_model` ayant enregistré les évolutions les plus importantes au trimestre validé, pour le groupe sélectionné dans l'historique, ici **Gamma Credit | IT-Vie** : `Written_Premium`, `Earned_Premium`, `Commission`, `Policy_count` et `Sinistres_attendus_ELR`.

Pour identifier les variables ayant le plus évolué, la valeur actuelle est comparée à la médiane des trimestres précédents. Cette variation est standardisée par une mesure robuste de dispersion, afin de pouvoir comparer des variables d'échelles différentes :

```math
z=xactuel−meˊdiane(xpasseˊ)max⁡(eˊcart interquartile(xpasseˊ) ; 1 %×∣meˊdiane(xpasseˊ)∣)
```

Le classement repose ensuite sur la valeur absolue de $`z`$.

Le principal résultat est le **contraste entre les variables explicatives et la cible**. Au Q4 2024, `Written_Premium`, `Earned_Premium`, `Commission` et `Policy_count` progressent de manière modérée, d'environ **4 à 6 %** par rapport à leur médiane passée. `Sinistres_attendus_ELR`, qui traduit les sinistres attendus au vu des primes et du loss ratio historique, n'augmente que d'environ **7 %**. Dans le même temps, la cible `Claims_incurred` du groupe augmente d'environ **75 %**.

Ce contraste est particulièrement important pour l'interprétation de l'anomalie détectée. L'activité du portefeuille et les sinistres attendus restent relativement stables, alors que les sinistres observés connaissent une forte augmentation. Si les principaux moteurs de la cible évoluent peu, la cible n'aurait pas dû bondir : son niveau au trimestre validé mérite donc une attention particulière. Les variables explicatives permettent ainsi de replacer l'anomalie détectée dans le contexte économique et actuariel du sous-portefeuille concerné.

Il convient néanmoins de distinguer **association** et **causalité**. Le classement présenté mesure l'ampleur des mouvements des variables entre les périodes considérées : il ne permet pas, à lui seul, d'affirmer qu'une variable est responsable de l'évolution de `Claims_incurred`. Une analyse complémentaire serait nécessaire pour établir une relation causale.
