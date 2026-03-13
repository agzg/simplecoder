
import random
import json

class Game:
    def __init__(self):
        self.rooms = {
            'Whispering Corridor': {
                'description': "You are in a damp, stone corridor. The torches on the walls flicker, casting long, dancing shadows. You hear faint, unsettling whispers that seem to come from the very walls around you. Passages lead NORTH to the library and EAST to a bathroom. A dark curtain hangs to the WEST.",
                'exits': {'north': 'Hogwarts Library', 'west': 'Restricted Section', 'east': "Moaning Myrtle\'s Bathroom", 'south': 'Slytherin\'s Study'}
            },
            'Hogwarts Library': {
                'description': "You are in the main hall of the Hogwarts Library. Thousands of books line towering shelves, and the air is thick with the smell of old parchment and dust. Most students have fled, leaving the room eerily silent. A passage leads SOUTH back to the corridor. You notice a heavy, leather-bound book left open on a table.",
                'exits': {'south': 'Whispering Corridor'}
            },
            'Restricted Section': {
                'description': "You slip behind a dark curtain into the Restricted Section. The air here is colder, and the books seem to watch you with malevolent intent. Chains bind many of them to the shelves. A single, unlit TORCH rests in a sconce on the far wall. The only way out is EAST.",
                'exits': {'east': 'Whispering Corridor'},
                'items': ['torch']
            },
            "Moaning Myrtle\'s Bathroom": {
                'description': "You enter a disused girls\' bathroom, flooded in several places. The constant sound of dripping water echoes off the tiles. In the center of the room is a row of sinks, one of which has a tiny snake carved into the tap. Lying glinting in a puddle on the floor is an ANCIENT KEY. A strange gurgling sound comes from a large pipe, suggesting a way DOWN.",
                'exits': {'down': 'The Chamber of Secrets'},
                'items': ['key']
            },
            'The Chamber of Secrets': {
                'description': "You slide down a long, dark pipe and land in a vast, green-lit chamber. Huge stone pillars carved with serpents rise to a ceiling lost in shadow. At the far end of the chamber is a colossal statue of a wizard\'s face, and set into the wall beside it is a heavy stone DOOR with a large, ornate lock. The sound of something immense slithering in the darkness makes your blood run cold.",
                'exits': {}
            },
            'Slytherin\'s Study': {
                'description': "You enter a dark, circular room. The walls are lined with bookcases filled with ancient, leather-bound tomes. A large, ornate desk sits in the center of the room, and a fire crackles in the hearth, casting dancing shadows on the walls. The air is thick with the smell of old books and something else... something cloying and sweet, like old blood.",
                'exits': {'north': 'Whispering Corridor'}
            }
        }
        self.player = {
            'current_room': 'Whispering Corridor',
            'inventory': [],
            'health': 1,
            'moves': 0,
            'entered_wrong_library': False
        }
        self.game_over = False

    def get_highest_score(self):
        try:
            with open('highscore.json', 'r') as f:
                data = json.load(f)
                return data['score'], data['name']
        except (FileNotFoundError, json.JSONDecodeError):
            return float('inf'), None
        
    def save_high_score(self, score, name):
        with open('highscore.json', 'w') as f:
            json.dump({'score': score, 'name': name}, f)

    def play(self):
        while not self.game_over:
            self.show_status()
            command = input('> ').lower().split()
            if not command:
                continue
            
            action = command[0]
            if action == 'go':
                self.go(command[1] if len(command) > 1 else '')
            elif action == 'get':
                self.get(command[1] if len(command) > 1 else '')
            elif action == 'use':
                self.use(command[1] if len(command) > 1 else '')
            elif action == 'look':
                self.look()
            elif action == 'quit':
                self.game_over = True
            else:
                print('Invalid command.')
            
            if not self.game_over:
                self.player['moves'] += 1


    def show_status(self):
        room = self.rooms[self.player['current_room']]
        print(f'\n--- {self.player["current_room"]} ---')
        print(room['description'])
        if 'items' in room and room['items']:
            print(f'You see: {", ".join(room["items"])}')
        print(f'Exits: {", ".join(room["exits"].keys())}')
        print(f'Inventory: {self.player["inventory"]}')


    def go(self, direction):
        room = self.rooms[self.player['current_room']]
        if direction in room['exits']:
            self.player['current_room'] = room['exits'][direction]
            if self.player['current_room'] == 'The Chamber of Secrets':
                self.serpent_encounter()
            elif self.player['current_room'] == 'Hogwarts Library':
                self.voldemort_at_the_library()
            elif self.player['current_room'] == 'Slytherin\'s Study':
                self.voldemort_encounter()
        else:
            print('You can\'t go that way.')

    def get(self, item_name):
        room = self.rooms[self.player['current_room']]
        if 'items' in room and item_name in room['items']:
            self.player['inventory'].append(item_name)
            room['items'].remove(item_name)
            print(f'You picked up the {item_name}.')
        else:
            print(f'There is no {item_name} here.')

    def use(self, item_name):
        if self.player['current_room'] == 'The Chamber of Secrets' and item_name == 'key':
            if 'key' in self.player['inventory']:
                print("You thrust the Ancient Key into the lock. With a deafening grind of stone, the heavy door slides open, revealing a passage to freedom. You scramble through and seal the Chamber behind you, trapping the monster forever. YOU HAVE ESCAPED! CONGRATULATIONS!")
                self.game_over = True
            else:
                print("You don't have the key.")
        else:
            print("You can't use that here.")

    def look(self):
        if self.player['current_room'] == 'Hogwarts Library':
            print("You read the page. It's from *Moste Potente Potions* but a note is scribbled in the margin: 'The Serpent of Slytherin fears the rooster\'s crow, but shies from bright, dancing flames.'")
        else:
            print('There is nothing special to look at.')

    def serpent_encounter(self):
        if 'torch' not in self.player['inventory']:
            print('From the shadows, a monstrous serpent lunges towards you with blinding speed!')
            if random.randint(1, 3) == 1:
                print('Its cavernous mouth closes around you. The last thing you see are its giant, yellow fangs. You have been eaten by the Serpent of Slytherin. GAME OVER.')
                self.player['health'] = 0
                self.game_over = True
            else:
                print("You dive out of the way just in time! The serpent\'s massive head smashes into the pillar behind you, showering you with stone chips. It slithers back into the shadows, preparing for another strike. You must act fast!")
        else:
            print("A monstrous serpent, its scales shimmering like emeralds, emerges from the shadows! It opens its mouth to strike, but as you raise your TORCH, the beast recoils from the bright flame. It hisses in frustration and retreats into the darkness, giving you a chance to act.")

    def voldemort_at_the_library(self):
        if not self.player['entered_wrong_library']:
            self.player['entered_wrong_library'] = True
            if random.randint(1, 10) == 1:
                print("Voldemort appears! With a flick of his wand, he casts the Killing Curse. A flash of green light is the last thing you see. GAME OVER.")
                self.player['health'] = 0
                self.game_over = True
            else:
                print("You feel a chill run down your spine, but the presence fades. You were lucky this time.")

    def voldemort_encounter(self):
        # This is for Slytherin\'s Study, let\'s make it a game over
        print("You have entered the study of the dark lord himself. Before you can react, you are struck by a curse. GAME OVER.")
        self.player['health'] = 0
        self.game_over = True

if __name__ == '__main__':
    Game().play()
