# Game Design Document

## Theme / Setting
Vampire Fantasy / Twilight  

## Player's Goal
The player has to avoid getting hunted by vampires and become one to win

## Locations (4-6)
Forks High School (Mix of vampires and humans)
The Cullen House (Vampires live here)
The Black House (Werewolves live here)
Bella's House 
The Woods (Vampires and Werewolves) 

```
           [Cullen House]
                |
[Bella's House]--+--[Forks High School]--[Black House]
                |
            [The Woods]
           (High Danger)
```

## Enemies (2-4 types)
[Describe your enemy types and their stats/behaviors]
Friendly Vampires - they want to suck your blood but try not to (you never know what they'll decide) 
Mean Vampires - they suck your blood every chance they get 
Werewolves - they attack you viciously 

## Win Condition
[How does the player win?]
The player wins when she becomes a vampire and marries her vampire love interest 

## Lose Condition
[How does the player lose?]
The player loses if one of the enemies kills her 

## Class Hierarchy

Character (base class)
├── Player
│    └── Bella
├── NPC
│    ├── Vampire (base)
│    │    ├── FriendlyVampire
│    │    └── MeanVampire
│    └── Werewolf

Location (base class)
├── ForksHighSchool
├── CullenHouse
├── BlackHouse
├── BellasHouse
└── Woods

Game (controller class)

## Additional Notes
[Any other design decisions, ideas, or plans]
