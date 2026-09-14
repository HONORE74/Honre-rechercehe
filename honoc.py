Explication sur le Dashboard
Les graphiques du Dashboard sont construits à l'aide de la bibliothèque Plotly, notamment avec les objets de type FigureWidget. Ces objets permettent de créer des graphiques interactifs pouvant être actualisés en fonction des interactions avec le Dashboard.
Plusieurs fonctions sont utilisées pour construire les différents éléments graphiques. Par exemple, la fonction _creer_cercle() construit le graphique circulaire, tandis que la fonction _creer_barres() construit le graphique en barres. Ces fonctions créent d'abord un objet Figure, puis celui-ci est transformé en FigureWidget afin de pouvoir être intégré au Dashboard.
L'affichage est ensuite réalisé à l'aide de la fonction display(). Dans le code final, plusieurs appels à display() sont effectués successivement :
display(fw_cercle)
display(_bandeau(...))
display(z_cartes)
display(fw_barres)
display(fw_forest)
...
Ainsi, chaque appel à display() affiche successivement un élément du Dashboard : graphique, bandeau, carte ou tableau. C'est la succession de ces appels qui permet de construire l'affichage complet du Dashboard dans Jupyter.
Les cinq graphiques apparaissent donc ensemble parce qu'ils sont tous appelés dans la même séquence d'affichage. Ils ne constituent pas nécessairement un seul graphique : ce sont plusieurs objets graphiques indépendants qui sont affichés successivement dans la même interface Jupyter.
Par ailleurs, les éléments interactifs du Dashboard sont construits avec ipywidgets. Les différents widgets, graphiques et tableaux sont ainsi combinés pour former une interface unique.
Et surtout, je corrigerais une phrase de ton texte actuel
Tu écris :
« Cependant la succession sous forme de dashboard est possible car jupyter permet en effet de compiler les graphiques. »
Je ne garderais pas cette formulation, parce que techniquement, « compiler les graphiques » n'est pas vraiment ce que fait Jupyter.
Je mettrais plutôt :
« La succession des différents éléments permet ainsi de construire l'affichage du Dashboard directement dans l'environnement Jupyter. »
C'est beaucoup plus précis.
Pour répondre directement à la question de ton tuteur
S'il te demande :
« Pourquoi les 5 graphiques sortent ensemble ? »
Tu peux lui répondre très simplement :
« Les cinq graphiques sont des objets distincts, mais ils sont tous appelés successivement avec display() dans le même environnement Jupyter. Jupyter les affiche donc les uns à la suite des autres, avec les autres éléments du Dashboard comme les bandeaux, les cartes et les tableaux. »
