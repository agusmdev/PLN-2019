## PLN-2019 - Práctico 1


### Ejercicio 1: Corpus

Para la realización de todo el proyecto se utilizó un corpus formado por libros en español de la saga Game of Thrones.
Los primeros 5 libros, se juntaron en un solo archivo "corpus.txt".
Para testear utilicé el sexto libro de la saga,
en el archivo "test.txt".
En la notebook create_corpus.ipynb está el procedimiento llevado a cabo para pasar de pdf a texto, con algunos preprocesados simples del texto.


### Ejercicio 2: Modelo de Ngramas

Se implementaron las funciones necesarias para el correcto funcionamiento del modelo y se agregó una funcion `tag_sentence()` utilizada para agregar _n-1_ __<s>__ tags de comienzo de oracion y el tag final de oración __</s>__.


### Ejercicio 3: Generación de texto

Dado un modelo ya entrenado, podemos generar oraciones basadas en las probabilidades de que se genere una palabra, dadas las n-1 palabras anteriores. Los siguientes son ejemplos de oraciones generadas con modelos de unigramas, bigramas, trigramas y cuatrigramas:

* Unigrama:

 * los . , tienen dedos por

 * raíces el Roose a 

 * que sonido una jubón sabrosos , beber — me ni 
   dijo el junto madre no 

 * guante Has niños enorme . a y nunca repetido — 
   de mejor Hierro capucha del tuviera hierba 
   buen Sansa acantilado en acercaron a 
 
 * vuestras allí ser víctima príncipe los ... 

* Bigrama:
 * — dijo Meñique tuviera un cinturón . 

 * Al fondo del arroyo para ver esa pesada . 

 * Una princesa lo digo yo somos un lobo y buena chica obedeció ... ¿ Qué estupidez . 

 * La niña con nadie . 

 * El Sueño del mar salado — Entonces a los edificios que tú . 

* Trigrama:

 * Jaime se estiró para ver esto — les gritó Arya . 

 * Coge su espada al servicio de Lord Arryn ? 

 * Me comprometí a contraer matrimonio . 

 * Martin Tormenta de espadas II celemín con manos ávidas . 

 * Rápida como una capa gruesa con cuello de Prendahl y Sallor el Calvo y Prendahl na Ghezn rodaron por las ventanas altas y puntiagudas daban al callejón en el mundo dice — respondió como hacía cada vez más , tanto que no se parecían un recuerdo de aquel primer brindis e hizo una mueca a Jon fuera de la montaña — dijo —. 

* Cuatrigrama:

 * Esos pentoshis se beberían sus orines si fueran tintos . 

 * Así que Varys tenía razón . 

 * Lord Tywin no parecía preocupado —. 

 * Tráeme el frasco , Clydas . 

 * — Pero vos no erais tan timorato , claro . 

### Ejercicio 4: Suavizado AddOne

Para la implementacion de este modelo, heredamos casi todos los métodos de la clase NGram, modificando algunos y sumando otro `V()` utilizado para calcular la longitud del vocabulario y poder realizar el suavizado

### Ejercicio 5: Evaluación de los modelos

En este ejercicio se evalúa la perplexity y cross-entropy y log-probability de los modelos con suavizado.
A continuacion listamos cada modelo para n-grams de tamaño {1,2,3,4} con sus perplexities respectivamente:

* AddOne:
 * 1479.56
 * 3060.65
 * 19621.81
 * 32981.55

* Interpolated:
 * 1614.22
 * 492.78
 * 443.52
 * 439.84

* Backoff:
 * 1614.22
 * 368.41
 * 316.16
 * 313.19


### Ejercicio 6: Suavizado por Interpolación

Se implementó la clase InterpolatedNGram. Heredando los metodos de la clase NGram y modificando algunos de ellos. Para calcular la probabilidad condicional se utilizó la formula explicada en las notas complementarias: 
https://wiki.cs.famaf.unc.edu.ar/lib/exe/fetch.php?media=materias:pln:2019:lm-notas.pdf

Para encontrar el gamma, me di cuenta que entre el rango(1, 20) elevando en cada iteracion i^2 se encontraba el gamma optimo que minimiza la perplexity.


### Ejercicio 7: Suavizado por Back-Off con Discounting

Se implementó la clase BackOffNGram, heredando los metodos de NGram y modificando algunos.

Para calcular la probabilidad condicional de este modelo se utiliza la forma general de backoff con discounting, tal como se explica en las notas complementarias linkeadas anteriormente.
