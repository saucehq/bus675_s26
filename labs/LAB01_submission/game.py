"""
Lab 1: Text-Based Adventure RPG
================================
Alexxis Saucedo 

Build your game here! This file contains all the starter code from the lab notebook.
Fill in the TODOs, add your own classes, and make it your own.

Run with: python game.py
"""

import random


# =============================================================================
# Dice Utilities
# =============================================================================

def roll_d20():
    """Roll a 20-sided die."""
    return random.randint(1, 20)


def roll_dice(num_dice, sides):
    """Roll multiple dice and return the total. E.g., roll_dice(2, 6) for 2d6."""
    return sum(random.randint(1, sides) for _ in range(num_dice))


# =============================================================================
# Character Classes
# =============================================================================

class Character:
    """Base class for all characters in the game."""

    def __init__(self, name, health, strength, defense):
        self.name = name 
        self.max_health = health 
        self.health = health 
        self.strength = strength 
        self.defense = defense

    def is_alive(self):
        return self.health > 0

    def take_damage(self, amount):
        self.health -= amount
        if self.health < 0:
            self.health = 0 
        print(f"{self.name} takes {amount} damage! (HP:{self.health}/{self.max_health})")

        if not self.is_alive():
            print(f"{self.name} has been killed!")
  

    def attack(self, target):
        print(f"\n{self.name} attacks {target.name}!")

        roll = roll_d20()
        attack_total = roll + self.strength
        print(f"Attack roll: {roll} + {self.strength} = {attack_total} vs DEF {target.defense}")
        crit = (roll == 20)

        if crit or attack_total >= target.defense:
            damage = roll_dice(1, 8) + self.strength
            if crit:
                damage += roll_dice(1, 8)
                print("CRITICAL HIT!")
            print(f"Hit! {self.name} deals {damage} damage.")
            target.take_damage(damage)
        else:
            print(f"Miss! {self.name} fails to hit {target.name}.")

    def __str__(self):
        return f"{self.name} (HP: {self.health}/{self.max_health})"
class Player(Character):
    """The player character."""
        # TODO: Call parent __init__ with appropriate starting stats
    def __init__(self, name):
        super().__init__(name=name, health=30, strength=2, defense=12)
        self.inventory = []
        self.vampire_progress = 0 
    
    def pick_up(self, item):
        self.inventory.append(item)
        print(f"{self.name} picked up: {item}")

    def show_inventory(self):
        print("\nInventory:")
        if not self.inventory:
            print(" (empty)")
        else: 
            for i, item in enumerate(self.inventory, start=1):
                print(f" {i}. {item}")

class Enemy(Character):
    """Base class for enemies."""

    def __init__(self, name, health, strength, defense, xp_value=10):
        super().__init__(name=name, health=health, strength=strength, defense=defense)
        self.xp_value = xp_value

class FriendlyVampire(Enemy):
    def __init__(self):
        super().__init__("Friendly Vampire", health=22, strength=4, defense=8, xp_value=10)

    def attack(self, target):
        if random.random() < 0.40:
                print(f"|n{self.name} hesitates... and does NOT attack.")
                return
        super().attack(target)
class MeanVampire(Enemy):
    def __init__(self):
        super().__init__("Mean Vampire", health=26, strength=5, defense=12, xp_value=15)
class Werewolf(Enemy):
    def __init__(self):
        super().__init__("Werewolf", health=28, strength=6, defense=12, xp_value=20)
    
    def attack(self, target):
        print(f"|n{self.name} lunges viciously!")
        super().attack(target)       

# =============================================================================
# Location Class
# =============================================================================

class Location:
    """A location in the game world."""

    def __init__(self, name, description, danger=0.2):
        self.name = name
        self.description = description
        self.danger = danger
        self.connections = {}  # {"north": Location, "south": Location, etc.}
        self.enemies = []      # List of enemies in this location
        self.items = []        # List of items in this location

    def add_connection(self, direction, location):
        self.connections[direction] = location

    def get_exits(self):
        return list(self.connections.keys())

    def describe(self):
        """Print a full description of the location."""
        print(f"\n{'='*50}")
        print(f"📍 {self.name}")
        print(f"{'='*50}")
        print(self.description)
        
        #enemies
        living = [e for e in self.enemies if e.is_alive()]
        if living:
            print("\n Enemies here:")
            for e in living:
                print(f"  - {e}")
        else:
            print("\n No enemies here.")

        # items
        if self.items:
            print("\n Items here:")
            for item in self.items:
                print(f"  - {item}")
        else:
            print("\n No items here.")

        # exits
        exits = self.get_exits()
        if exits:
            print("\n Exits:")
            for d in exits:
                print(f"  - {d} → {self.connections[d].name}")
        else:
            print("\n No exits available.")

# =============================================================================
# World Builder
# =============================================================================

def create_world():
    forks_high = Location(
        "Forks High School",
        "A school in a small, rainy town. Everything is not as it seems.",
        danger=0.25
    )

    cullen_house = Location(
        "Cullen House",
        "Too perfect. Something must be off.",
        danger=0.20
    )

    black_house = Location(
        "Black House",
        "Feels warm.",
        danger=0.35
    )

    bellas_house = Location(
        "Bella's House",
        "Home. Safe (I think). Charlie is there.",
        danger=0.10
    )

    woods = Location(
        "The Woods",
        "Dark, cold, and wet. Danger is all around.",
        danger=0.60
    )

    # Connections
    forks_high.add_connection("west", bellas_house)
    forks_high.add_connection("east", black_house)
    forks_high.add_connection("north", cullen_house)
    forks_high.add_connection("south", woods)

    bellas_house.add_connection("east", forks_high)
    black_house.add_connection("west", forks_high)
    cullen_house.add_connection("south", forks_high)
    woods.add_connection("north", forks_high)

    # Items
    bellas_house.items = ["Phone", "Pepper Spray"]
    forks_high.items = ["Bandage"]
    black_house.items = ["Wolf Totem"]
    cullen_house.items = ["Cullen Ring"]
    woods.items = ["Mystery Blood Vial"]

    # Enemies
    forks_high.enemies = [FriendlyVampire()]
    woods.enemies = [MeanVampire()]
    black_house.enemies = [Werewolf()]

    locations = {
        "Forks High School": forks_high,
        "Cullen House": cullen_house,
        "Black House": black_house,
        "Bella's House": bellas_house,
        "The Woods": woods,
    }

    start_location = forks_high
    return locations, start_location

# =============================================================================
# Combat System
# =============================================================================

class Combat:
    """Manages turn-based combat between player and enemy."""

    # Combat states
    PLAYER_TURN = "player_turn"
    ENEMY_TURN = "enemy_turn"
    COMBAT_END = "combat_end"

    def __init__(self, player, enemy):
        self.player = player
        self.enemy = enemy
        self.state = Combat.PLAYER_TURN
        self.combat_log = []

    def start(self):
        """Begin combat and run until someone wins/loses/flees."""
        print(f"\n COMBAT BEGINS! ")
        print(f"{self.player.name} vs {self.enemy.name}!")

        while self.state != Combat.COMBAT_END:
            if self.state == Combat.PLAYER_TURN:
                self.player_turn()
            elif self.state == Combat.ENEMY_TURN:
                self.enemy_turn()

        return self.get_result()

    def player_turn(self):
        """Handle player's turn in combat."""
        print(f"\n{self.player} | {self.enemy}")
        print("What do you do? (attack / run)")

        action = input("> ").lower().strip()

        if action == "attack":
            self.player.attack(self.enemy)
            if not self.enemy.is_alive():
                print(f"\n {self.enemy.name} has been defeated!")
                self.state = Combat.COMBAT_END
            else:
                self.state = Combat.ENEMY_TURN

        elif action == "run":
            # 50% chance to escape
            if random.random() < 0.5:
                print("You successfully fled!")
                self.state = Combat.COMBAT_END
            else:
                print("Couldn't escape!")
                self.state = Combat.ENEMY_TURN

        else:
            print("Invalid action. Try 'attack' or 'run'.")

    def enemy_turn(self):
        """Handle enemy's turn in combat."""
        print(f"\n{self.enemy.name}'s turn...")
        self.enemy.attack(self.player)

        if not self.player.is_alive():
            print(f"\n {self.player.name} has fallen!")
            self.state = Combat.COMBAT_END
        else:
            self.state = Combat.PLAYER_TURN

    def get_result(self):
        """Return the combat result: 'victory', 'defeat', or 'fled'."""
        if not self.enemy.is_alive():
            return "victory"
        elif not self.player.is_alive():
            return "defeat"
        else:
            return "fled"


# =============================================================================
# Main Game Class
# =============================================================================

class Game:
    """Main game controller."""

    # Game states
    EXPLORING = "exploring"
    GAME_OVER = "game_over"
    VICTORY = "victory"

    def __init__(self):
        self.player = None
        self.locations = {}
        self.current_location = None
        self.state = Game.EXPLORING
        self.game_running = True

    def start(self):
        """Initialize and start the game."""
        self.show_intro()
        self.create_player()
        self.locations, self.current_location = create_world()  # Your function from earlier
        self.current_location.describe()

        # Main game loop
        while self.game_running:
            if self.state == Game.EXPLORING:
                self.exploration_loop()
            elif self.state == Game.GAME_OVER:
                self.show_game_over()
                break
            elif self.state == Game.VICTORY:
                self.show_victory()
                break

    def show_intro(self):
        """Display the game introduction."""
        print("\n" + "="*60)
    print("\nThe rain never stops in Forks, Washington.")
    print("You’ve just arrived to start a new life… but something feels off.")
    print("Pale students who never eat. Golden eyes watching from the woods.")
    print("Low growls echoing at night.")

    print("\nRumors whisper of vampires and werewolves.")
    print("If you survive long enough, you might become one of them.")
    print("Find the Cullen Ring. Embrace immortality.")
    print("Or die trying.")

    print("\n Survive. Transform. Marry your vampire love interest.")
    print("=" * 60)

    def create_player(self):
        """Create the player character."""
        print("\nWhat is your name, adventurer?")
        name = input("> ")
        self.player = Player(name)
        print(f"\nWelcome, {name}! Your adventure begins...")


    def exploration_loop(self):
        print("\nWhat do you do? (type 'help' for commands)")
        command = input("> ").lower().strip()

        parts = command.split()
        if not parts:
            return

        action = parts[0]

        if action == "help":
            self.show_help()

        elif action == "look":
            self.current_location.describe()

        elif action == "go" and len(parts) > 1:
            self.move(parts[1])

        elif action in ["north", "south", "east", "west"]:
            self.move(action)

        elif action in ["fight", "attack"]:
            self.initiate_combat()

        elif action in ["inventory", "i"]:
            self.player.show_inventory()

        elif action == "take" and len(parts) > 1:
            item_name = " ".join(parts[1:])
            self.take_item(item_name)

        elif action == "quit":
            print("Thanks for playing!")
            self.game_running = False

        else:
            print("I don't understand that command. Type 'help' for options.")

        self.check_victory()

    def move(self, direction):
        if direction in self.current_location.connections:
            self.current_location = self.current_location.connections[direction]
            self.current_location.describe()
        else:
            print(f"You can't go {direction} from here.")

    def take_item(self, item_name):
        # case-insensitive match
        for item in list(self.current_location.items):
            if item.lower() == item_name.lower():
                self.current_location.items.remove(item)
                self.player.pick_up(item)
                return
        print("That item isn't here.")

    def initiate_combat(self):
        if not self.current_location.enemies:
            print("There's nothing to fight here.")
            return

        enemy = self.current_location.enemies[0]
        battle = Combat(self.player, enemy)
        result = battle.start()

        if result == "victory":
            self.current_location.enemies.remove(enemy)
            # small vampire progress reward for surviving fights
            self.player.vampire_progress = min(100, self.player.vampire_progress + 20)
            print(f" Vampire Progress: {self.player.vampire_progress}/100")
        elif result == "defeat":
            self.state = Game.GAME_OVER

    def check_victory(self):
        # Win: become vampire + have Cullen Ring
        if self.player.vampire_progress >= 100 and "Cullen Ring" in self.player.inventory:
            self.state = Game.VICTORY

    def show_help(self):
        print("\n AVAILABLE COMMANDS:")
        print("  go [direction] - Move (north, south, east, west)")
        print("  look          - Examine surroundings")
        print("  fight         - Attack an enemy here")
        print("  take [item]   - Pick up an item here")
        print("  inventory (i) - Check inventory")
        print("  help          - Show commands")
        print("  quit          - Exit game")
        print("\n WIN: Vampire Progress >= 100 AND you have the Cullen Ring.")
        print(" LOSE: Your HP hits 0.")

    def show_game_over(self):
        print("\n" + "=" * 60)
        print("                    GAME OVER")
        print("=" * 60)
        print("\nYou have fallen. The adventure ends here...")
        print("\n(But you can always try again!)")

    def show_victory(self):
        print("\n" + "=" * 60)
        print("                     VICTORY! 💍")
        print("=" * 60)
        print("\nYou become a vampire and slip on the Cullen Ring...")
        print("You marry your vampire love interest. Forks will never be the same.")




# =============================================================================
# Run the Game
# =============================================================================

if __name__ == "__main__":
    game = Game()
    game.start()
