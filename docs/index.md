# Unity RL Controller (`urc`)

Librería + CLI para controlar entrenamientos de Reinforcement Learning en Unity desde la
terminal: bridge Unity↔código, algoritmos, mapas, hiperparámetros, evaluación y visualización,
todo con comandos simples y componentes intercambiables.

**Idea central**: todo tiene un valor por defecto que funciona sin configurar nada, y todo se
puede reemplazar sin tocar el core.

| Eje | Por defecto | Personalizable |
|---|---|---|
| Conexión con el entorno (**bridge**) | Unity ML-Agents | Socket TCP, subproceso, o cualquier lenguaje propio (ver [protocolo](protocol.md)) |
| Algoritmo de entrenamiento | PPO de Stable-Baselines3 | SAC, o tu propio algoritmo como plugin de Python |
| Entorno/mapa | Lo que declares en `urc.yaml` | Parámetros, currículo, domain randomization |

## Por dónde empezar

- **[Guía rápida](quickstart.md)** — instalación y tu primer entrenamiento, sin necesitar Unity.
- **[Referencia de comandos](cli-reference.md)** — todos los comandos de `urc`.
- **[Ejemplos](examples.md)** — proyectos completos: sin Unity, con un bridge en C#, y con Unity real.
- **[Tutoriales](tutorials/write-a-bridge.md)** — cómo extender cada pieza intercambiable.

## Diseño

El diseño completo, los principios y el histórico de decisiones (con sus porqués) están en el
**[Roadmap](roadmap.md)**. Si vas a modificar `urc` o quieres entender por qué algo se hizo de una
forma y no de otra, es la lectura de referencia.
