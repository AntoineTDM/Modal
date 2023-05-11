# Modal





Tentative pour centraliser le code du projet final

Question à résoudre: 
Comment maximiser la data augmentation sur la première phase du projet?
Comment éviter l'overfitting lorsqu'on entraine sur beaucoup d'epoch avec les mêmes images? Peut on diviser le set en 5 par exemple et tester sur un cinquième?
Avons nous un testing set?

Plan d'action:
1. Data augmentation
2. Entrainer un premier modele adoc 
3. Rajouter les images qui sont "presque sures" avec le modele 1 aux catégories correspondantes 
4. Reprendre les étapes 1-3 en controlant la qualité des images rajoutées et en boostant le modèle. 
5. Tester différents modèles ad hoc 
6. Tenter de générer nous même des images 



Questions à poser demain:
Comment sauver un model (ou ses weights) pour ne pas avoir à le relecharger / réentrainer à chaque fois?


