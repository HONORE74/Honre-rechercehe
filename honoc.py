# Détection et interprétation des anomalies

## 1. Pourquoi considérons-nous certaines observations comme des anomalies ?

L'objectif de cette analyse est de déterminer si une observation réalisée au trimestre considéré est compatible avec ce que le modèle pouvait raisonnablement attendre, compte tenu des informations disponibles au moment de la prédiction.

Une prédiction n'est pas une valeur que l'observation réelle doit nécessairement reproduire exactement. Le modèle fournit une estimation de la valeur attendue à partir des variables explicatives et des informations historiques disponibles lors de son entraînement. Il existe donc naturellement une certaine variabilité entre la valeur prédite et la valeur réellement observée.

C'est dans ce contexte que les **Conformal Predictions** deviennent particulièrement intéressantes. Elles permettent de ne pas se limiter à une valeur centrale prédite, mais d'associer à cette prédiction un **intervalle de valeurs considéré comme compatible avec le comportement attendu**.

La question n'est donc pas simplement :

> *L'observation est-elle exactement égale à la prédiction ?*

La question est plutôt :

> *L'observation reste-t-elle compatible avec ce que le modèle considère comme une évolution normale ?*

Une observation située à l'intérieur de l'intervalle de prédiction reste compatible avec la variabilité attendue par le modèle. À l'inverse, lorsqu'elle se situe en dehors de cet intervalle, elle constitue une **anomalie potentielle** et mérite une analyse complémentaire.

Il est important de préciser qu'une sortie de l'intervalle ne signifie pas automatiquement que la donnée est erronée. Elle signifie d'abord que l'observation est inhabituelle au regard de ce que le modèle avait prévu.

**[IMAGE À INSÉRER ICI — Historique des prédictions avec les valeurs observées et les intervalles de prédiction]**

---

## 2. Rôle de l'intervalle de prédiction

Pour chaque prédiction, le modèle fournit une valeur centrale ainsi qu'un intervalle de prédiction.

On peut représenter cette situation de manière simplifiée :

$$
[\text{borne basse},\text{borne haute}]
$$

avec une prédiction centrale :

$$
\hat{y}
$$

et une valeur réellement observée :

$$
y_{obs}
$$

Les intervalles de prédiction sont construits avec une **couverture cible de 90 %**, correspondant à un niveau de référence de $1-\alpha$. Cette couverture est utilisée pour apprécier les observations situées en dehors des intervalles attendus.

Tant que l'observation se situe dans l'intervalle, elle reste compatible avec la variabilité attendue par le modèle.

À l'inverse, lorsque l'observation sort de cet intervalle, elle constitue une anomalie potentielle.

**[IMAGE À INSÉRER ICI — Schéma montrant la prédiction centrale, l'intervalle conforme et une observation à l'intérieur puis à l'extérieur de l'intervalle]**

---

## 3. Présentation du tableau de bord

Le tableau de bord porte sur l'indicateur `Claims_incurred` et présente les résultats de la détection des observations atypiques pour le quatrième trimestre 2024 (Q4 2024).

Cette période correspond au trimestre retenu pour la validation des prédictions issues du modèle. Les différents éléments du tableau de bord sont ainsi centrés sur cette période, à l'exception des représentations consacrées à l'évolution historique de l'indicateur.

Pour le Q4 2024, le tableau de bord porte sur **135 sous-portefeuilles prédits**. Parmi les observations analysées, **16 anomalies** sont identifiées.

Ces anomalies correspondent aux lignes pour lesquelles le `score_composite` est strictement positif. Ce score est utilisé pour caractériser le degré d'atypicité des observations selon les critères retenus dans l'approche de détection.

Ainsi, le tableau de bord permet de passer d'une vision globale des prédictions à une identification ciblée des observations nécessitant une attention particulière.

La présence de 16 anomalies parmi les 135 sous-portefeuilles analysés fournit notamment un premier aperçu de l'étendue des observations atypiques détectées sur le périmètre du Q4 2024.

**[IMAGE À INSÉRER ICI — Vue générale du tableau de bord]**

---

# 4. Interprétation des cartes indicateurs

Les cartes indicateurs fournissent une synthèse des principaux résultats obtenus sur le périmètre étudié pour le Q4 2024.

L'analyse porte sur **135 sous-portefeuilles**, correspondant à un total de **2 700 lignes dans `df_model`**, soit 135 sous-portefeuilles observés sur 20 trimestres.

Sur ce périmètre, **16 anomalies sont détectées**, soit **11,9 %** des 135 sous-portefeuilles analysés.

Ce résultat indique qu'une proportion limitée du périmètre présente un comportement identifié comme atypique selon le score de détection retenu.

La valeur affichée ne correspond donc pas à la totalité des observations de `df_model`, mais au nombre de sous-portefeuilles classés comme anomalies pour la période considérée.

La période d'analyse correspond au Q4 2024, qui constitue la période validée par le modèle.

Les statistiques associées à la variable étudiée montrent une moyenne de **973 044**, avec des valeurs comprises entre **−732 296 et 5 452 841**.

L'écart important entre le minimum et le maximum traduit une dispersion marquée des valeurs observées sur le périmètre.

La présence d'une valeur minimale négative est également visible dans les résultats et correspond, d'après le tableau de bord, à l'erreur de signe identifiée comme **F**.

Le classement permet ensuite d'identifier l'observation présentant le niveau d'anomalie le plus élevé sur le périmètre.

Le rang n°1 est associé au sous-portefeuille :

**Gamma Credit | IT - Vie | Protection | Disability**

Cette information permet ainsi de cibler les observations qui nécessitent une attention particulière dans le cadre de l'analyse des anomalies.

Enfin, la couverture observée est de **87,4 %**, soit **118 sous-portefeuilles sur 135** dont la valeur observée se situe à l'intérieur de l'intervalle de prédiction.

Cette valeur est inférieure à la couverture cible de 90 % définie pour le modèle. Il existe donc **17 sous-portefeuilles situés en dehors de l'intervalle de prédiction**.

La couverture constitue ainsi un indicateur important pour apprécier l'adéquation entre les intervalles produits par le modèle et les observations effectivement réalisées.

Cette couverture inférieure à la cible de 90 % signifie que les observations sortant des intervalles sont plus nombreuses que ce qui serait attendu avec une couverture de 90 %.

Elle constitue donc un signal à examiner dans l'analyse de la qualité des intervalles et de la présence éventuelle de comportements atypiques.

Toutefois, la couverture seule ne permet pas d'attribuer automatiquement ces écarts à des anomalies réelles. Cette identification repose sur les critères complémentaires utilisés dans la procédure de détection, notamment le `score_composite`.

**[IMAGE À INSÉRER ICI — Cartes indicateurs du tableau de bord]**

---

# 5. État des sous-portefeuilles

La figure présente l'état des **135 sous-portefeuilles analysés pour le Q4 2024**.

Chaque barre représente un sous-portefeuille et les observations sont ordonnées selon la hiérarchie :

**Partner > Companies > Lob > Risk**

Cette organisation permet de conserver la structure du portefeuille tout en facilitant la localisation des observations atypiques.

La distinction entre les deux couleurs permet d'identifier rapidement les sous-portefeuilles concernés par une anomalie.

Les barres vertes correspondent aux sous-portefeuilles pour lesquels aucune anomalie n'a été détectée, tandis que les barres rouges correspondent à ceux dont le `score_composite` est strictement positif.

Sur l'ensemble du périmètre, **16 anomalies** sont ainsi identifiées parmi les 135 sous-portefeuilles, soit **11,9 %** du périmètre.

La disposition des barres rouges permet également d'étudier la répartition des anomalies au sein du portefeuille.

Certaines apparaissent de manière isolée, alors que d'autres sont relativement proches les unes des autres. Cette proximité peut constituer un premier indice permettant de rechercher des caractéristiques communes entre les sous-portefeuilles concernés, notamment au niveau du partenaire ou de la compagnie.

Elle ne permet toutefois pas, à elle seule, d'établir l'existence d'une cause commune. Une analyse complémentaire des caractéristiques des observations concernées reste nécessaire pour confirmer cette éventuelle relation.

Cette représentation apporte ainsi une information complémentaire au nombre global d'anomalies présenté précédemment.

Alors que la carte indicateur permet de quantifier les anomalies détectées, cette visualisation permet de les situer dans la structure du portefeuille.

Enfin, lorsque le nombre de sous-portefeuilles devient supérieur à 160, les barres sont regroupées par tranches afin de préserver la lisibilité de la représentation. Cette modification concerne uniquement le mode d'affichage et n'affecte pas le principe de détection des anomalies.

**[IMAGE À INSÉRER ICI — État des 135 sous-portefeuilles avec les observations normales et atypiques]**

---

# 6. Concentration des anomalies par secteur

Le graphique circulaire permet d'analyser la répartition des anomalies selon les niveaux **Partner** et **Companies**.

Les secteurs représentés correspondent aux regroupements définis par la structure du portefeuille. La taille de chaque secteur est proportionnelle à sa contribution à la somme des `score_composite` observés sur l'ensemble du périmètre.

Le résultat met en évidence une forte concentration du score d'anomalie sur le secteur **Gamma Credit**, qui représente à lui seul **87 % de la somme des `score_composite`** du périmètre.

Cette proportion élevée s'explique notamment par la présence, dans ce secteur, de l'anomalie présentant le score le plus important.

Le tableau de bord indique en effet un score de **472 199,72** pour la première anomalie, classée au rang #1.

Cette représentation doit toutefois être interprétée avec précaution.

Le fait que Gamma Credit représente 87 % du score total ne signifie pas que 87 % des anomalies appartiennent à ce secteur.

Il s'agit ici d'une mesure de la contribution au score d'anomalie total. Autrement dit, les anomalies rattachées à Gamma Credit concentrent une part très importante de l'intensité mesurée par le `score_composite`.

La couleur des secteurs apporte une information complémentaire. Elle correspond au `score_composite` de la première anomalie du secteur, avec une intensité croissante allant du bleu vers le rouge.

Ainsi, une couleur plus proche du rouge traduit la présence d'une anomalie ayant un score élevé au sein du secteur considéré.

Dans le cas de Gamma Credit, la forte intensité du secteur est cohérente avec la présence de l'anomalie classée n°1, dont le score atteint **472 199,72**.

Cette visualisation permet donc de compléter l'analyse réalisée précédemment.

Alors que la répartition des sous-portefeuilles permettait d'identifier les emplacements des anomalies, le graphique circulaire met en évidence leur concentration en termes d'intensité du score.

**[IMAGE À INSÉRER ICI — Graphique circulaire de concentration des anomalies par secteur]**

---

# 7. Analyse du cercle à la maille la plus fine

Cette représentation propose une lecture hiérarchique des anomalies en descendant progressivement dans les différentes dimensions du portefeuille :

**Partner → Companies → Lob → Risk**

Chaque niveau du cercle correspond ainsi à une maille d'analyse plus fine. L'anneau extérieur permet finalement d'identifier les anomalies individuellement au niveau du sous-portefeuille.

L'intérêt principal de cette représentation est de permettre de localiser précisément une anomalie au sein de la structure du portefeuille.

En partant d'un partenaire, l'analyse peut être affinée successivement jusqu'à la compagnie, puis au segment d'activité (`Lob`) et enfin au risque (`Risk`).

Cette décomposition permet donc de passer d'une vision globale de la concentration des anomalies à l'identification du sous-portefeuille auquel elles sont rattachées.

La figure met notamment en évidence la concentration observée autour de **Gamma Credit**, puis permet de poursuivre l'analyse à des niveaux plus détaillés.

À la maille la plus fine, chaque secteur extérieur correspond à une anomalie individuelle.

Le score associé à l'anomalie est représenté par la couleur du secteur, selon une intensité allant du bleu vers le rouge. Une couleur plus proche du rouge correspond ainsi à un `score_composite` plus élevé.

Cette représentation est particulièrement utile pour la phase d'investigation.

Lorsqu'un secteur présentant une anomalie est identifié, il est possible de remonter dans la hiérarchie afin de déterminer précisément quel partenaire, quelle compagnie, quel segment d'activité et quel risque sont concernés.

L'utilisateur peut ensuite sélectionner le secteur correspondant afin de filtrer le reste du tableau de bord sur le périmètre concerné.

**[IMAGE À INSÉRER ICI — Cercle hiérarchique à la maille la plus fine]**

---

# 8. Identification des observations atypiques

Le graphique présente les **12 premières observations atypiques**, classées selon leur niveau d'anomalie, de la plus élevée à la moins élevée.

Chaque barre correspond ainsi à une observation identifiée par le processus de détection et rattachée à un sous-portefeuille donné.

La longueur de la barre représente le montant observé de `Claims_incurred` pour l'observation considérée.

Elle permet donc de comparer directement les montants observés entre les différentes anomalies.

La majorité des valeurs représentées sont positives, mais une observation se distingue par une valeur négative : **Beta Retail | FR-Vie**, classée en deuxième position.

Sa barre se situe à gauche de zéro, ce qui traduit un montant négatif pour l'indicateur étudié.

La couleur des barres apporte une information différente. Elle correspond au `score_composite`, avec une intensité plus importante lorsque le niveau d'anomalie est élevé.

Ainsi, l'observation classée n°1, **Gamma Credit | IT-Vie**, apparaît en rouge, ce qui traduit le score d'anomalie le plus élevé du périmètre présenté.

Son `score_composite` est de **472 199,72**.

Cette distinction entre la longueur et la couleur est importante pour l'interprétation du graphique.

Une valeur élevée de `Claims_incurred` ne signifie pas nécessairement qu'une observation constitue l'anomalie la plus importante.

Le classement dépend du `score_composite`, qui tient compte de l'écart entre l'observation et ce qui était attendu selon le modèle et son intervalle de prédiction.

Ainsi, une observation peut présenter un montant élevé tout en ayant un score d'anomalie inférieur à celui d'une autre observation.

Le graphique permet également d'observer qu'un même libellé peut apparaître plusieurs fois.

C'est notamment le cas de **Gamma Credit | IT-Vie**, présent aux rangs #1 et #7.

Ces deux occurrences correspondent à deux observations distinctes, même si elles sont rattachées au même groupe.

Enfin, le survol d'une barre permet d'obtenir des informations complémentaires sur l'observation sélectionnée, notamment sa valeur observée, sa valeur prédite, son intervalle de prédiction et son `score_composite`.

**[IMAGE À INSÉRER ICI — Top 12 des observations atypiques]**

---

# 9. Interprétation des intervalles de prédiction conformes

Le forest plot présente les **12 premières observations atypiques**, dans le même ordre que celui utilisé dans le graphique précédent.

Pour chaque sous-portefeuille, trois éléments principaux sont représentés :

- la bande bleue correspond à l'intervalle de prédiction conforme à une couverture cible de 90 % ;
- le losange représente la valeur prédite par le modèle ;
- le point rouge correspond à la valeur effectivement observée de `Claims_incurred`.

La comparaison entre la valeur observée et l'intervalle de prédiction permet d'identifier visuellement les observations qui s'écartent de la plage attendue.

Lorsqu'un point rouge se situe en dehors de la bande bleue, la valeur effectivement observée n'est pas contenue dans l'intervalle de prédiction associé.

Ces observations constituent ainsi des cas particulièrement intéressants pour la procédure de détection des comportements atypiques.

La ligne pointillée reliant la valeur prédite à la valeur observée permet de visualiser directement l'écart entre ces deux valeurs.

Plus cet écart est important relativement à l'intervalle de prédiction, plus l'observation apparaît éloignée de ce qui était attendu par le modèle.

Cette représentation permet donc de visualiser simultanément la prédiction, l'incertitude associée à cette prédiction et la réalisation effectivement observée.

L'observation **Gamma Credit | IT-Vie**, classée première dans les graphiques précédents, illustre particulièrement bien cette situation.

La valeur observée est nettement supérieure à l'intervalle de prédiction, ce qui traduit un écart important entre le montant effectivement réalisé et la plage de valeurs attendue par le modèle.

Cette observation est également associée au `score_composite` le plus élevé, de **472 199,72**, ce qui explique son classement en première position parmi les anomalies présentées.

À l'inverse, certaines observations présentent des écarts moins importants.

C'est notamment le cas de **Epsilon Auto | IT-Vie**, classée #8.

La valeur observée se situe légèrement au-delà de la bande de prédiction, alors même que le montant de `Claims_incurred` est relativement élevé.

Cette situation illustre un point important : l'importance du montant observé ne suffit pas à déterminer le niveau d'anomalie.

Une valeur élevée peut rester relativement proche de l'intervalle attendu compte tenu du niveau d'incertitude du modèle.

**[IMAGE À INSÉRER ICI — Forest plot des 12 premières anomalies]**

---

# 10. Évolution historique et identification de l'anomalie

Le premier graphique présente l'évolution de l'indicateur `Claims_incurred` pour le groupe **Delta Conso | FR-NonVie**, sur les dix derniers trimestres disponibles, de **2022-Q3 à 2024-Q4**.

La série permet ainsi d'observer le comportement historique de ce groupe avant et pendant la période de validation.

Sur la majeure partie de la période, les valeurs évoluent dans une plage relativement proche, avec quelques fluctuations entre les trimestres.

On observe notamment une hausse en 2023-Q4, suivie d'une diminution en 2024-Q1 et 2024-Q2, puis d'une nouvelle progression en 2024-Q3.

Le dernier trimestre, **2024-Q4**, se distingue cependant par une augmentation importante de `Claims_incurred`.

La valeur observée associée à cette dernière période est de **5 258 053**.

Elle constitue le point le plus élevé de la série présentée et se distingue nettement des niveaux observés au cours des trimestres précédents.

Cette évolution constitue donc un premier signal d'écart par rapport au comportement historique du groupe.

Il est toutefois important de distinguer l'augmentation historique de la détection d'une anomalie.

Le fait que la dernière valeur soit particulièrement élevée ne suffit pas, à lui seul, à conclure qu'il s'agit d'une anomalie.

Pour déterminer si cette observation est effectivement atypique, il faut la comparer à la valeur attendue par le modèle ainsi qu'à l'incertitude associée à cette prédiction.

**[IMAGE À INSÉRER ICI — Évolution historique de Claims_incurred de 2022-Q3 à 2024-Q4]**

---

# 11. Analyse à la maille la plus fine : prédiction et intervalle

Le second graphique affine l'analyse en considérant le sous-portefeuille :

**Delta Conso | FR-NonVie | Creditor | Unemployment**

Il reprend l'évolution de `Claims_incurred` sur la période historique et ajoute, pour le Q4 2024, les éléments issus du modèle de prédiction.

Pour les trimestres historiques, la courbe représente les valeurs effectivement observées.

Au Q4 2024, plusieurs informations sont désormais disponibles simultanément :

- le losange représente la valeur prédite par le modèle ;
- la bande correspond à l'intervalle de prédiction conforme à 90 % ;
- le point rouge représente la valeur effectivement observée.

La comparaison entre ces éléments met en évidence un écart important.

La valeur observée au Q4 2024, de **5 258 053**, se situe nettement au-dessus de la valeur prédite et en dehors de l'intervalle de prédiction conforme.

L'observation réalisée est donc très éloignée de la plage de valeurs attendue par le modèle pour ce sous-portefeuille.

Cette représentation permet de comprendre pourquoi cette observation attire l'attention du dispositif de détection.

L'anomalie ne résulte pas uniquement du fait que `Claims_incurred` atteint un niveau élevé.

Elle résulte surtout du fait que le niveau effectivement observé est nettement supérieur à ce que le modèle avait anticipé, au regard de l'intervalle d'incertitude associé à la prédiction.

L'intérêt de cette analyse à la maille la plus fine est également de pouvoir rattacher précisément l'écart observé à ses différentes dimensions :

**Partner, Companies, Lob et Risk.**

Dans le cas présenté, l'analyse conduit ainsi jusqu'au sous-portefeuille **Delta Conso | FR-NonVie | Creditor | Unemployment**, ce qui facilite l'identification du périmètre concerné et les éventuelles investigations complémentaires.

**[IMAGE À INSÉRER ICI — Historique + prédiction + intervalle à la maille la plus fine]**

---

# 12. Les trois critères de tri à la maille la plus fine

À la maille la plus fine, le tableau de bord permet de sélectionner un sous-portefeuille parmi les observations identifiées.

La liste présentée sur la figure regroupe les sous-portefeuilles concernés et, par défaut, ceux-ci sont classés selon le `score_composite`.

Le graphique associé se met alors à jour en fonction de l'observation sélectionnée, sans modifier les autres éléments du tableau de bord.

Trois modes de classement sont proposés :

1. le `score_composite` ;
2. l'écart à l'intervalle ;
3. le montant observé.

Chacun permet d'aborder les anomalies sous un angle différent.

### 12.1. Classement par `score_composite`

Le premier critère est le `score_composite`, qui constitue le classement privilégié dans le tableau de bord.

Il permet de hiérarchiser les anomalies en combinant la gravité relative de l'écart et l'enjeu associé à l'observation.

Ce classement permet ainsi de ne pas considérer uniquement l'ampleur brute de l'écart.

Une anomalie présentant un écart important relativement à son propre niveau d'incertitude peut être davantage mise en évidence qu'une observation présentant un montant élevé mais dont le comportement reste relativement cohérent avec l'intervalle attendu.

Dans la liste affichée, **Gamma Credit | IT-Vie | Protection | Disability**, avec un montant observé de **5 346 514**, apparaît ainsi en première position lorsque le tri est effectué selon le `score_composite`.

### 12.2. Classement par écart à l'intervalle

Le deuxième critère repose sur l'écart à l'intervalle de prédiction.

Il mesure dans quelle proportion l'observation se situe au-delà de l'intervalle attendu, relativement à la largeur de cet intervalle.

Ce critère permet donc de mettre en évidence les dépassements les plus nets, indépendamment de l'enjeu associé au montant observé.

Il peut notamment faire ressortir des anomalies sur des sous-portefeuilles de faible montant, dès lors que la valeur observée s'éloigne fortement de la plage attendue.

Ce classement apporte ainsi une lecture complémentaire au `score_composite` : il répond davantage à la question de savoir **à quel point l'observation s'écarte de son intervalle**, plutôt qu'à celle de savoir quelle anomalie représente l'enjeu global le plus important.

### 12.3. Classement par montant observé

Enfin, le tableau de bord permet de classer les observations selon leur montant observé de `Claims_incurred`.

Ce mode de tri permet de mettre en avant les observations présentant les montants les plus élevés et apporte donc une lecture davantage orientée vers la matérialité.

Cependant, un montant élevé ne signifie pas nécessairement qu'il s'agit de l'observation la plus atypique.

Une valeur importante peut être cohérente avec le comportement historique du sous-portefeuille et avec l'intervalle de prédiction qui lui est associé.

À l'inverse, une valeur plus faible peut constituer une anomalie importante si elle s'écarte fortement de ce qui était attendu.

**[IMAGE À INSÉRER ICI — Interface permettant de choisir les trois critères de tri]**

---

# 13. Analyse des variables explicatives

Cette figure présente les cinq variables numériques de `df_model` qui ont le plus évolué au Q4 2024 pour le groupe **Gamma Credit | IT-Vie** :

- `Written_Premium`
- `Earned_Premium`
- `Commission`
- `Policy_count`
- `Sinistres_attendus_EUR`

La comparaison est réalisée par rapport à la médiane des trimestres précédents, afin d'identifier les évolutions inhabituelles tout en tenant compte des différences d'échelle entre les variables.

Le principal résultat est le contraste entre les différentes variables.

Les `Written_Premium`, `Earned_Premium`, `Commission` et `Policy_count` évoluent de manière relativement modérée, avec une progression d'environ **5 % au Q4 2024**.

En revanche, la variable `Sinistres_attendus_EUR` augmente de **75 %**, soit une évolution nettement plus importante que celle des autres indicateurs.

Ce contraste est particulièrement important pour l'interprétation de l'anomalie détectée.

Alors que l'activité du portefeuille reste relativement stable, les sinistres attendus connaissent une forte augmentation.

Cette évolution atypique peut donc contribuer à expliquer le comportement inhabituel de la cible au trimestre validé.

Les variables explicatives permettent ainsi de replacer l'anomalie détectée dans le contexte économique et actuariel du sous-portefeuille concerné.

Enfin, il faut rester prudent dans l'interprétation : cette analyse met en évidence une **association, et non une causalité**.

Elle permet d'identifier les variables qui ont le plus évolué, mais ne permet pas d'affirmer qu'une variable est directement responsable de l'évolution de `Claims_incurred`.

**[IMAGE À INSÉRER ICI — Cinq variables explicatives et leur évolution au Q4 2024]**

---

# 14. Mesurer l'importance de l'écart

Le fait qu'une observation soit située en dehors de l'intervalle ne suffit pas nécessairement à déterminer son niveau de criticité.

Deux observations peuvent toutes les deux être situées en dehors de l'intervalle, mais l'une peut être très proche de la borne tandis que l'autre peut en être très éloignée.

C'est pourquoi deux composantes sont utilisées pour caractériser l'écart observé.

## 14.1. Écart par rapport à l'intervalle

La première composante mesure à quelle distance l'observation se situe de la borne de l'intervalle, relativement à la largeur de celui-ci.

On définit :

$$
A =
\frac{|y_{obs}-\text{borne franchie}|}
{\text{largeur de l'intervalle}}
$$

Par exemple, avec un intervalle :

$$
[80,120]
$$

la largeur de l'intervalle est :

$$
120-80=40
$$

Si l'observation vaut :

$$
y_{obs}=121
$$

alors :

$$
A =
\frac{121-120}{40}
=
0{,}025
$$

L'observation est donc seulement légèrement située au-delà de la borne.

Si, au contraire :

$$
y_{obs}=150
$$

alors :

$$
A =
\frac{150-120}{40}
=
0{,}75
$$

L'écart par rapport à l'intervalle est alors beaucoup plus important.

**[IMAGE À INSÉRER ICI — Exemple visuel du calcul de A]**

---

## 14.2. Écart par rapport à la prédiction

La deuxième composante mesure l'écart entre l'observation et la prédiction centrale.

Elle est définie par :

$$
B =
\frac{|y_{obs}-\hat{y}|}
{|\hat{y}|}
$$

Cette mesure permet d'exprimer l'écart relativement à la valeur prédite.

Par exemple, si :

$$
\hat{y}=100
$$

et :

$$
y_{obs}=105
$$

alors :

$$
B =
\frac{|105-100|}{100}
=
5\%
$$

L'écart reste donc relativement faible.

En revanche, si :

$$
y_{obs}=160
$$

alors :

$$
B =
\frac{|160-100|}{100}
=
60\%
$$

L'écart par rapport à la prédiction est alors beaucoup plus important.

**[IMAGE À INSÉRER ICI — Exemple visuel du calcul de B]**

---

# 15. Construction du score de priorisation

Les deux composantes précédentes sont ensuite combinées afin de construire un score permettant de prioriser les observations à examiner.

Le score est défini comme suit :

$$
score_{composite}
=
norm(A\times B)\times GWP
$$

où :

- $A$ mesure l'écart de l'observation par rapport à l'intervalle ;
- $B$ mesure l'écart relatif entre l'observation et la prédiction ;
- $GWP$ permet de prendre en compte le niveau habituel du sous-portefeuille ;
- $norm(A\times B)$ correspond à la normalisation du produit $A\times B$.

Le produit $A\times B$ devient particulièrement élevé lorsque l'observation est **à la fois éloignée de sa borne et éloignée de la prédiction**.

Cette combinaison permet donc de distinguer une simple sortie légèrement au-delà de l'intervalle d'une observation présentant un écart beaucoup plus important.

**[IMAGE À INSÉRER ICI — Formule ou schéma présentant les composantes du score composite]**

---

# 16. Pourquoi le GWP intervient-il dans le score ?

Le niveau du sous-portefeuille doit également être pris en compte dans la priorisation.

À gravité comparable, une anomalie observée sur un portefeuille représentant un enjeu financier important peut nécessiter davantage d'attention qu'une anomalie portant sur un portefeuille de faible poids.

Le GWP permet ainsi d'intégrer une dimension liée à **l'importance économique du sous-portefeuille**.

Le score ne mesure donc pas uniquement l'écart statistique. Il permet également de tenir compte de l'enjeu associé à l'observation.

---

# 17. Pourquoi une sortie de l'intervalle ne signifie-t-elle pas nécessairement une erreur ?

Une observation située en dehors de l'intervalle conforme doit être considérée comme une **anomalie potentielle**, et non comme une erreur certaine.

Plusieurs situations sont possibles.

Une anomalie peut correspondre à une véritable évolution du portefeuille, à un événement exceptionnel ou à une évolution inhabituelle mais réelle.

Elle peut également révéler un problème dans les données : erreur de saisie, problème d'unité, doublon, mauvais rattachement, décalage de période ou autre incohérence.

L'analyse doit donc permettre de distinguer :

> **une observation inhabituelle mais correcte**

de

> **une observation inhabituelle résultant d'une erreur dans les données.**

Si la vérification confirme qu'il s'agit d'un événement réel, par exemple un sinistre important, l'observation n'a pas nécessairement vocation à être corrigée.

Elle doit plutôt être **documentée et expliquée**.

---

# 18. Sens de l'écart et conséquence financière

Le sens dans lequel l'observation sort de l'intervalle apporte également une information importante.

## Observation sous la borne basse

Lorsque :

$$
y_{obs}<\text{borne basse}
$$

les sinistres enregistrés sont inférieurs à ce que le modèle attendait.

Cela peut notamment correspondre à une sous-estimation des flux manquants, à un décalage de période ou à une erreur de signe.

Dans un contexte actuariel, une sous-estimation peut conduire à des **provisions insuffisantes**, avec un risque financier futur.

## Observation au-dessus de la borne haute

Lorsque :

$$
y_{obs}>\text{borne haute}
$$

les sinistres enregistrés sont supérieurs à ce que le modèle attendait.

Cette situation peut notamment être liée à un doublon, une erreur d'unité ou un mauvais rattachement.

Elle peut conduire à une **mobilisation excessive de provisions ou de capital** si la donnée est effectivement erronée.

**[IMAGE À INSÉRER ICI — Schéma présentant une sortie sous la borne basse et une sortie au-dessus de la borne haute]**

---

# 19. Pourquoi ce n'est ni une simple erreur du modèle, ni une simple erreur d'intervalle, ni nécessairement une évolution normale ?

Le `score_composite` combine trois dimensions afin de caractériser plus précisément les observations signalées :

$$
score_{composite}
=
norm(A\times B)\times GWP
$$

avec :

$$
A =
\frac{|y_{obs}-\text{borne franchie}|}
{\text{largeur de l'intervalle}}
$$

et :

$$
B =
\frac{|y_{obs}-y_{préd}|}
{|y_{préd}|}
$$

Cette combinaison permet d'écarter plusieurs explications concurrentes.

### A : sortie de l'intervalle

Cette composante mesure de combien l'observation dépasse la borne franchie, relativement à la largeur de l'intervalle.

Elle permet donc de distinguer une sortie très faible de l'intervalle d'une sortie beaucoup plus importante.

### B : erreur relative

Cette composante mesure l'écart entre l'observation et la prédiction, rapporté à la prédiction.

Elle permet ainsi de vérifier que l'observation est également éloignée de la valeur attendue par le modèle.

### GWP : niveau habituel du sous-portefeuille

Le GWP permet de prendre en compte le niveau habituel du sous-portefeuille et donc son enjeu économique.

Le produit $A\times B$ n'est élevé que si l'observation est **à la fois loin de sa borne et loin de la prédiction**.

Une simple sortie de quelques pourcents de la largeur de l'intervalle peut donc rester relativement peu importante.

De même, une erreur de prédiction modérée ne suffit pas nécessairement à produire un score élevé si l'observation reste dans une zone compatible avec l'intervalle.

**[IMAGE À INSÉRER ICI — Schéma des trois composantes A, B et GWP]**

---

# 20. Exemple de lecture de A et B

Supposons :

$$
\hat{y}=100
$$

et un intervalle de prédiction :

$$
[80,120]
$$

### Cas 1 : observation à 121

On obtient :

$$
A=
\frac{121-120}{40}
=
0{,}025
$$

et :

$$
B=
\frac{|121-100|}{100}
=
21\%
$$

L'observation est légèrement en dehors de l'intervalle et son écart par rapport à la prédiction reste relativement limité.

### Cas 2 : observation à 150

On obtient :

$$
A=
\frac{150-120}{40}
=
0{,}75
$$

et :

$$
B=
\frac{|150-100|}{100}
=
50\%
$$

Dans ce second cas, l'observation est beaucoup plus éloignée à la fois de l'intervalle et de la prédiction.

Cette comparaison illustre pourquoi toutes les sorties d'intervalle ne présentent pas nécessairement le même niveau d'anomalie.

**[IMAGE À INSÉRER ICI — Comparaison graphique des deux situations]**

---

# 21. Synthèse de l'analyse

Une observation est donc signalée parce qu'elle **sort de l'intervalle conforme** construit par les Conformal Predictions.

Comme une couverture de 90 % laisse théoriquement attendre environ 10 % de sorties, une sortie d'intervalle constitue d'abord une **anomalie potentielle** et non une anomalie certaine.

Le `score_composite` permet ensuite de confirmer et de hiérarchiser les observations en tenant compte de plusieurs dimensions :

- l'écart par rapport à l'intervalle ;
- l'écart relatif par rapport à la prédiction ;
- l'enjeu économique du sous-portefeuille.

L'analyse doit ensuite être complétée par une lecture métier.

Le sens de l'écart apporte notamment une information sur le risque financier potentiel : une observation sous la borne peut correspondre à une sous-estimation, tandis qu'une observation au-dessus de la borne peut correspondre à une surestimation.

Dans les deux cas, une vérification est nécessaire afin de déterminer si l'observation correspond à une véritable évolution du portefeuille ou à une anomalie dans les données.

L'objectif final n'est donc pas de dire automatiquement qu'une observation est fausse, mais de **détecter les situations qui nécessitent une vérification et une interprétation métier**.

---

# 22. Conclusion

L'approche mise en place permet ainsi de passer progressivement :

**de la prédiction → à l'intervalle de prédiction → à la détection des sorties d'intervalle → à la priorisation des anomalies → puis à leur interprétation métier.**

Les Conformal Predictions apportent une zone de compatibilité autour de la prédiction.

Le `score_composite` permet ensuite de hiérarchiser les observations selon l'importance de leur écart et l'enjeu associé.

Enfin, l'analyse des variables explicatives et de l'historique permet de replacer les observations signalées dans leur contexte économique et actuariel.

L'analyse constitue ainsi un outil d'aide à l'investigation : elle permet de concentrer l'attention sur les observations les plus atypiques tout en laissant l'interprétation finale à l'analyse métier et à la vérification des données.
