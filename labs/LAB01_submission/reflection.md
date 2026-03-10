# Reflection: OOP Design Decisions

Write 2-3 paragraphs reflecting on your object-oriented design. Some questions to consider:

- Why did you structure your classes the way you did?
- What inheritance relationships did you use and why?
- What was challenging about managing multiple interacting objects?
- If you had more time, what would you refactor or add?
- How does this experience connect to working with OOP in analytics/ML codebases?

---

I decided to structure the player as Bella Swan to stay true to the film (for the film lovers). I then added mean vampires and friendly vampires to even out th enemies and again stay true to the story, not all the vampires wanted to kill Bella. I also added werewolves as enemies as well because they are also a major part of the Twilight world. 

To organize the game, I used object oriented programming so different parts of the game each had their own role. The character base class was created so that the player and enemies share things like health, attacking, and taking damage. I then created subclasses of Player, Enemy, FriendlyVampire, MeanVampire and Werewolf to give each type slightly different behavior while still reusing the same core logic. I also used location class to represent different places in Forks, these store description, items, exits and enemies. 

One challenge was managing how all the objects interact with each other during gameplay. Bella moves between Location objects, which may contain enemies or items, and then the Combat system takes over when a fight begins. Making sure the game state updates correctly after combat, item pickups, or movement was tricky. If I had more time, I would add more locations, more enemies, and possibly more story interactions so the world feels bigger. This experience connects to working with object oriented code in machine learning projects because those projects also involve multiple components that interact with each other (datasets, models, and evaluation tools). Structuring code into clear classes helps make the system easier to understand (and debug.)
