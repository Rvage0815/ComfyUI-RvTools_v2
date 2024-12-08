﻿import random
import os
import folder_paths
import csv

from ..core import CATEGORY, cstr

comfy_path = os.path.dirname(folder_paths.__file__)
rvt_path = os.path.join(os.path.dirname(__file__))

file_path = ''

csv_file_name_1 =  '0_1Woman.csv'
csv_file_name_2 =  '0_2Man.csv'
csv_file_name_3 =  '0_3Fictional_Character.csv'
csv_file_name_4 =  '0_4Humanoids.csv'
csv_file_name_5 =  '0_5Animals.csv'
csv_file_name_6 =  '0_6Vehicles.csv'


#based on the code from jice
def getfilename(folder):
    name = []
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
           name.append(filename[3:-4])
    return name

def select_random_line_from_csv_file(file, folder):
    chosen_lines = []
    for filename in os.listdir(folder):
        if filename.endswith(".csv") and filename[3:-4] == file:
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r', encoding="utf-8") as file:
                lines = file.readlines()
                if lines:
                    chosen_lines.append(random.choice(lines).strip())
    lines_chosed = "".join(chosen_lines)
    return lines_chosed

class RvText_RandomPrompt:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TEXT.value
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "create_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        # Check input folder
        prompt_path = os.path.join(comfy_path, "models")
        prompt_path = os.path.join(prompt_path, "wildcards")
        prompt_path = os.path.join(prompt_path, "RvTools_v2")
        prompt_path = os.path.join(prompt_path, "subjects")
        
        file_path = prompt_path

        if not os.path.exists(prompt_path.strip()):
            cstr(f'Creating Folder `{prompt_path.strip()}`.').warning.print()
            try:
                os.makedirs(prompt_path, exist_ok=True)

                create_file_1(prompt_path)
                create_file_2(prompt_path)
                create_file_3(prompt_path)
                create_file_4(prompt_path)
                create_file_5(prompt_path)
                create_file_6(prompt_path)


            except OSError as e:
                cstr(f"The path `{prompt_path}` could not be created! Is there write access?\n{e}").error.print()

        for filename in os.listdir(prompt_path):
              if filename.endswith(".csv"):
                 file_path = os.path.join(prompt_path, filename)
                 lines = []
                 with open(file_path, 'r', encoding="utf-8") as file:
                     try:
                         lines = file.readlines()
                     except:
                         continue

                 required[filename [3:-4]] = (["disabled"] + ["🎲random"] + lines, {"default": "disabled"})
        return {
            "required": required,

            "optional": {
                "seed": ("INT", {"forceInput": True, "default": 0, "min": 0, "max": 1125899906842624}),
            }
        }    
    def create_prompt(self, **kwargs):
       
        name_of_files = getfilename(file_path)
        concatenated_values = ""
        final_values = ""
        values = []
        values = [""] * len(name_of_files)

        for i, filename in enumerate(name_of_files):
            if kwargs.get(filename, 0) == "🎲random":
                    values[i] = select_random_line_from_csv_file(filename, file_path)
            else:      
                    values[i] = kwargs.get(filename, 0)
                    values[i] = values[i].strip()
        for value in values:
            if value != "disabled":
                    concatenated_values += value + ","
        print(f"➡️Prompt: {concatenated_values [:-1]}")
        final_values += concatenated_values [:-1] + "\n" 
        concatenated_values = ""

        final_values = final_values.strip()  
         
        return (final_values,)

def create_file_1(prompt_path):
    file_content = []
    file_content.append("a assassin woman")
    file_content.append("a assassin woman")
    file_content.append("a athletic woman")
    file_content.append("a beautiful woman")
    file_content.append("a biker woman")
    file_content.append("a charismatic woman")
    file_content.append("a dreamy woman")
    file_content.append("a elder woman")
    file_content.append("a elegant woman")
    file_content.append("a expressive woman")
    file_content.append("a fashion woman")
    file_content.append("a fitness woman")
    file_content.append("a glamorous woman")
    file_content.append("a gorgeous woman")
    file_content.append("a muscular woman")
    file_content.append("a mysterious woman")
    file_content.append("a office woman")
    file_content.append("a rocker woman")
    file_content.append("a sexy woman")
    file_content.append("a singer woman")
    file_content.append("a stylish woman")
    file_content.append("a voluptuous woman")
    file_content.append("a warrior woman")
    file_content.append("a woman")
    file_content.append("a worker woman")

    filepath = os.path.join(prompt_path, csv_file_name_1)

    cstr(f'Creating File `{filepath.strip()}`.').warning.print()
    
    with open(filepath, "w", encoding="utf-8", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write each line as a separate row in the CSV file
        for line in file_content:           
            try:
                csv_writer.writerow([line])    
            except:
                continue
        
    return ()
def create_file_2(prompt_path):
    file_content = []
    file_content.append("a assassin man")
    file_content.append("a athletic man")
    file_content.append("a beautiful man")
    file_content.append("a biker man")
    file_content.append("a charismatic man")
    file_content.append("a dreamy man")
    file_content.append("a elder man")
    file_content.append("a elegant man")
    file_content.append("a expressive man")
    file_content.append("a fashion man")
    file_content.append("a fitness man")
    file_content.append("a glamorous man")
    file_content.append("a gorgeous man")
    file_content.append("a muscular man")
    file_content.append("a mysterious man")
    file_content.append("a office man")
    file_content.append("a rocker man")
    file_content.append("a sexy man")
    file_content.append("a singer man")
    file_content.append("a stylish man")
    file_content.append("a voluptuous man")
    file_content.append("a warrior man")
    file_content.append("a man")
    file_content.append("a worker man")

    filepath = os.path.join(prompt_path, csv_file_name_2)

    cstr(f'Creating File `{filepath.strip()}`.').warning.print()
    
    with open(filepath, "w", encoding="utf-8", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write each line as a separate row in the CSV file
        for line in file_content:           
            try:
                csv_writer.writerow([line])    
            except:
                continue
        
    return ()
def create_file_3(prompt_path):
    file_content = []
    file_content.append("2B from Nier")
    file_content.append("9S from Nier")
    file_content.append("A2 from Nier")
    file_content.append("Abed Nadir")
    file_content.append("Aerith Gainsborough from final fantasy")
    file_content.append("Agent 47")
    file_content.append("Agent Dale Cooper from Twin Peaks")
    file_content.append("Agent Mulder")
    file_content.append("Agent Smith")
    file_content.append("Akuma from street fighter")
    file_content.append("Aladdin")
    file_content.append("Albus Dumbledore")
    file_content.append("Alicia Florrick")
    file_content.append("Aloy")
    file_content.append("Altair Ibn-La'Ahad")
    file_content.append("Alucard")
    file_content.append("Alvin")
    file_content.append("Amaterasu")
    file_content.append("Amphitrite")
    file_content.append("Anakin Skywalker")
    file_content.append("Anansi")
    file_content.append("Android")
    file_content.append("Angel")
    file_content.append("Anger from Inside Out")
    file_content.append("Anna from Frozen")
    file_content.append("Anthropomorphic -animal-")
    file_content.append("Anubis")
    file_content.append("Apollo")
    file_content.append("Apollo Belenus")
    file_content.append("Apollo Musagetes")
    file_content.append("Aragorn")
    file_content.append("Ariel from The Little Mermaid")
    file_content.append("Armored Angel")
    file_content.append("Artemis")
    file_content.append("Arthur Curry")
    file_content.append("Arthur Morgan")
    file_content.append("Arwen")
    file_content.append("Ash Ketchum")
    file_content.append("Astarte")
    file_content.append("Athena")
    file_content.append("Atlas")
    file_content.append("Austin Powers")
    file_content.append("Ayane Hiragana")
    file_content.append("Bagheera from The Jungle Book")
    file_content.append("Baloo from The Jungle Book")
    file_content.append("Barbarella")
    file_content.append("Barbie")
    file_content.append("Bard the Bowman")
    file_content.append("Bart Simpson")
    file_content.append("Basil from The Great Mouse Detective")
    file_content.append("Batman")
    file_content.append("BB-8")
    file_content.append("Beatrice Prior")
    file_content.append("Beetlejuice")
    file_content.append("Bella Swan")
    file_content.append("Bellatrix Lestrange")
    file_content.append("Belle from Beauty and the Beast")
    file_content.append("Bender from Futurama")
    file_content.append("Beorn")
    file_content.append("Betty Boop")
    file_content.append("Betty Draper")
    file_content.append("Bilbo Baggins")
    file_content.append("Black Widow")
    file_content.append("Blanka from Street Fighter")
    file_content.append("Boba Fett")
    file_content.append("Boromir")
    file_content.append("Brian Griffin")
    file_content.append("Brock from Pokemon")
    file_content.append("Buddha")
    file_content.append("Buffy Summers")
    file_content.append("Bugs Bunny")
    file_content.append("Bunny Girl")
    file_content.append("Buzz Lightyear from Toy Story")
    file_content.append("C-3PO")
    file_content.append("Cammy from street fighter")
    file_content.append("Captain America")
    file_content.append("Captain Dallas")
    file_content.append("Captain Hook")
    file_content.append("Captain Kirk")
    file_content.append("Captain Nemo")
    file_content.append("Carmela Soprano")
    file_content.append("Catwoman")
    file_content.append("Cayde-6")
    file_content.append("Celeborn")
    file_content.append("Ceres")
    file_content.append("Cernunnos")
    file_content.append("Cersei Lannister")
    file_content.append("Chakotay")
    file_content.append("Charlie Brown")
    file_content.append("Chell from half-life")
    file_content.append("Cheshire Cat")
    file_content.append("Chewbacca")
    file_content.append("Chun-Li from Street Fighter")
    file_content.append("Cinderella")
    file_content.append("Ciri from the witcher")
    file_content.append("Cloud Strife")
    file_content.append("Coatlicue")
    file_content.append("Cookie Monster")
    file_content.append("Count Dooku")
    file_content.append("Crash Bandicoot")
    file_content.append("Cupid")
    file_content.append("Cyborg")
    file_content.append("Daenerys Targaryen")
    file_content.append("Dagon")
    file_content.append("Dante")
    file_content.append("Darth Vader")
    file_content.append("Data from Star Trek")
    file_content.append("David 8")
    file_content.append("David Brent")
    file_content.append("Deadpool")
    file_content.append("Demeter")
    file_content.append("Demon Girl")
    file_content.append("Devil Girl")
    file_content.append("Dhalsim from Street Fighter")
    file_content.append("Dionysus")
    file_content.append("Doc Hudson from Cars")
    file_content.append("Doctor Evil")
    file_content.append("Donald Duck")
    file_content.append("Donkey from Shrek")
    file_content.append("Donkey Kong")
    file_content.append("Doomguy")
    file_content.append("Dr. House")
    file_content.append("Dr. Jekyll/Mr. Hyde")
    file_content.append("Draco Malfoy")
    file_content.append("Dracula")
    file_content.append("Edward Cullen")
    file_content.append("Edward Elric")
    file_content.append("Ellen Ripley")
    file_content.append("Elrond")
    file_content.append("Elsa from Frozen")
    file_content.append("Epimetheus")
    file_content.append("Ereshkigal")
    file_content.append("Eric Cartman")
    file_content.append("EVE from WALL-E")
    file_content.append("Evil Queen")
    file_content.append("Evil Witch")
    file_content.append("Fairy")
    file_content.append("Fiona from Shrek")
    file_content.append("Fortuna")
    file_content.append("Frankenstein's Monster")
    file_content.append("Freyja")
    file_content.append("Frodo Baggins")
    file_content.append("Fry from Futurama")
    file_content.append("Galadriel")
    file_content.append("Gandalf")
    file_content.append("Ganesh")
    file_content.append("George Weasley")
    file_content.append("Geralt of Rivia")
    file_content.append("Gimli")
    file_content.append("Ginny Weasley")
    file_content.append("GLaDOS")
    file_content.append("Glinda")
    file_content.append("Gollum")
    file_content.append("Goofy")
    file_content.append("Gordon Freeman from half-life")
    file_content.append("Goth Bride")
    file_content.append("Guybrush Threepwood")
    file_content.append("Gwen Stacy")
    file_content.append("Hades")
    file_content.append("Hagrid")
    file_content.append("Han Solo")
    file_content.append("Harley Quinn")
    file_content.append("Harry Potter")
    file_content.append("Hecate")
    file_content.append("Heihachi Mishima from tekken")
    file_content.append("Helios")
    file_content.append("Hello Kitty")
    file_content.append("Hera")
    file_content.append("Hermione Granger")
    file_content.append("Homer Simpson")
    file_content.append("Horus")
    file_content.append("HymenTriton")
    file_content.append("Hyperion")
    file_content.append("Inspector Gadget")
    file_content.append("Iron Man")
    file_content.append("Ivy Valentine")
    file_content.append("Izanami")
    file_content.append("Jack Sparrow")
    file_content.append("James Bond")
    file_content.append("Janus")
    file_content.append("Jean-Luc Picard")
    file_content.append("Jerry Mouse")
    file_content.append("Jesse Pinkman")
    file_content.append("Jessica Rabbit")
    file_content.append("Jill Valentine")
    file_content.append("Jin Kazama from tekken")
    file_content.append("John Wick")
    file_content.append("Katniss Everdeen")
    file_content.append("Ken Masters from street fighter")
    file_content.append("Kermit the Frog")
    file_content.append("Kratos")
    file_content.append("Kwan Yin")
    file_content.append("Kylo Ren")
    file_content.append("Lagertha from Vikings")
    file_content.append("Lara Croft")
    file_content.append("Legolas")
    file_content.append("Lieutenant Kane")
    file_content.append("Lightning from FF13")
    file_content.append("Lightning McQueen from Cars")
    file_content.append("Lilith")
    file_content.append("Loki")
    file_content.append("Lt. Ellen Ripley")
    file_content.append("Lucky Chloe from tekken")
    file_content.append("Luigi from Cars")
    file_content.append("Luke Skywalker")
    file_content.append("Max Payne")
    file_content.append("Medusa")
    file_content.append("Megaman")
    file_content.append("Mercury")
    file_content.append("Mickey Mouse")
    file_content.append("Minion")
    file_content.append("Miss Piggy")
    file_content.append("Misty from Pokemon")
    file_content.append("Mithras")
    file_content.append("Mona Lisa")
    file_content.append("Monkey D.Luffy")
    file_content.append("Morpheus")
    file_content.append("Morrigan")
    file_content.append("Mr. Potato Head from Toy Story")
    file_content.append("Mr. Spock")
    file_content.append("Mulan")
    file_content.append("Neo from the matix")
    file_content.append("Nero")
    file_content.append("Nurse Joy")
    file_content.append("nymph")
    file_content.append("nymphomaniac")
    file_content.append("Obi-Wan Kenobi")
    file_content.append("Odin")
    file_content.append("Osiris")
    file_content.append("Pacman")
    file_content.append("Padmé Amidala")
    file_content.append("Persephone")
    file_content.append("Peter Pan")
    file_content.append("Pikachu")
    file_content.append("Pink Panther")
    file_content.append("Pinocchio")
    file_content.append("Poison Ivy")
    file_content.append("Popeye")
    file_content.append("Poseidon")
    file_content.append("Princess Zelda")
    file_content.append("Prometheus")
    file_content.append("Proteus")
    file_content.append("Puss in Boots")
    file_content.append("Quetzalcoatl")
    file_content.append("Qui-Gon Jinn")
    file_content.append("R2-D2")
    file_content.append("Ra")
    file_content.append("Ragnar Lodbrok")
    file_content.append("Rapunzel")
    file_content.append("Rayne from BloodRayne")
    file_content.append("Remus Lupin")
    file_content.append("Remy from Ratatouille")
    file_content.append("Rey from star wars")
    file_content.append("Road Runner")
    file_content.append("Robin Hood")
    file_content.append("Sailor Moon")
    file_content.append("Saitama")
    file_content.append("Santa")
    file_content.append("Selene")
    file_content.append("Sherlock Holmes")
    file_content.append("Shiva")
    file_content.append("Shrek")
    file_content.append("Simba from The Lion King")
    file_content.append("Snow Queen")
    file_content.append("Snow White")
    file_content.append("Sonic the Hedgehog")
    file_content.append("Speedy Gonzales")
    file_content.append("Spider-Gwen")
    file_content.append("Spider-Man")
    file_content.append("Spiderman")
    file_content.append("Spider-Girl")
    file_content.append("SpongeBob SquarePants")
    file_content.append("Spyro the Dragon")
    file_content.append("Stormtrooper")
    file_content.append("Sub-Zero")
    file_content.append("Succubus")
    file_content.append("Super Saiyen")
    file_content.append("Superman")
    file_content.append("Tasmanian Devil")
    file_content.append("Temptress")
    file_content.append("Tezcatlipoca")
    file_content.append("The Alien Queen")
    file_content.append("The Brain")
    file_content.append("the bride from tim burton")
    file_content.append("The Grinch")
    file_content.append("The Horned God")
    file_content.append("The Hulk")
    file_content.append("The Joker")
    file_content.append("The Lorax")
    file_content.append("The Mandalorian")
    file_content.append("The Predators")
    file_content.append("The Terminator")
    file_content.append("The Xenomorphs")
    file_content.append("Thor")
    file_content.append("Tinker Bell")
    file_content.append("Trinity from the matrix")
    file_content.append("Tuvok from star trek")
    file_content.append("vampire")
    file_content.append("vampire princess")
    file_content.append("vampire queen")
    file_content.append("Vegeta")
    file_content.append("Venus")
    file_content.append("WALL-E")
    file_content.append("White Witch")
    file_content.append("Wile E. Coyote")
    file_content.append("Winnie the Pooh")
    file_content.append("Witch")
    file_content.append("Wonder Woman")
    file_content.append("Woodland Nymph")
    file_content.append("Yennefer of Vengerberg from the witcher")
    file_content.append("Yoda")
    file_content.append("Yosemite Sam")
    file_content.append("Yoshimitsu")
    file_content.append("Zeus")
    
    filepath = os.path.join(prompt_path, csv_file_name_3)

    cstr(f'Creating File `{filepath.strip()}`.').warning.print()
        
    with open(filepath, "w", encoding="utf-8", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write each line as a separate row in the CSV file
        for line in file_content:           
            try:
                csv_writer.writerow([line])    
            except:
                continue
        
    return ()
def create_file_4(prompt_path):
    file_content = []
    file_content.append("Aasimar")
    file_content.append("AI")
    file_content.append("Alien")
    file_content.append("Altmer")
    file_content.append("Android")
    file_content.append("Angel")
    file_content.append("Archangel")
    file_content.append("Argonian")
    file_content.append("Atronach")
    file_content.append("Automaton")
    file_content.append("Banshee")
    file_content.append("Beholder")
    file_content.append("Bene Gesserit")
    file_content.append("Bloodelf")
    file_content.append("Borg")
    file_content.append("Bosmer")
    file_content.append("Brownie")
    file_content.append("Cardassian")
    file_content.append("Centaur")
    file_content.append("Cherubim")
    file_content.append("Chimera")
    file_content.append("Cyberman")
    file_content.append("Cyborg")
    file_content.append("Cyclops")
    file_content.append("Cyclopskin")
    file_content.append("Cylon")
    file_content.append("Daedra")
    file_content.append("Dalek")
    file_content.append("Djinn")
    file_content.append("doll")
    file_content.append("Dracthyr")
    file_content.append("Draenei")
    file_content.append("Dragonborn")
    file_content.append("Dragonkin")
    file_content.append("Draugr")
    file_content.append("Drow")
    file_content.append("Dryad")
    file_content.append("Dunmer")
    file_content.append("Dwarf")
    file_content.append("Dwemer")
    file_content.append("Elemental")
    file_content.append("Elf")
    file_content.append("Faerie")
    file_content.append("Fairy")
    file_content.append("Falmer")
    file_content.append("Faun")
    file_content.append("Fremen")
    file_content.append("Frost giant")
    file_content.append("Garden gnome")
    file_content.append("Ghost")
    file_content.append("Giant")
    file_content.append("Gnome")
    file_content.append("Goblin")
    file_content.append("Godess")
    file_content.append("Golem")
    file_content.append("Gorgon")
    file_content.append("Gorn")
    file_content.append("Gremlin")
    file_content.append("Harpy")
    file_content.append("High elf")
    file_content.append("Hobbit")
    file_content.append("Homunculus")
    file_content.append("Ifrit")
    file_content.append("Jedi")
    file_content.append("Jinn")
    file_content.append("Khajit")
    file_content.append("Klingon")
    file_content.append("Lich")
    file_content.append("Lycanthrope")
    file_content.append("Mannequin")
    file_content.append("Martian")
    file_content.append("Mentat")
    file_content.append("Mermaid")
    file_content.append("Mind Flayer")
    file_content.append("Minotaur")
    file_content.append("Mummy")
    file_content.append("Murloc")
    file_content.append("Mutant")
    file_content.append("Na'vi")
    file_content.append("Naga")
    file_content.append("Necromancer")
    file_content.append("Nightelf")
    file_content.append("Nymph")
    file_content.append("Ogre")
    file_content.append("Orc")
    file_content.append("Orsimer")
    file_content.append("Pandaren")
    file_content.append("Pixie")
    file_content.append("Poltergeist")
    file_content.append("Predator")
    file_content.append("Redguard")
    file_content.append("Replicant")
    file_content.append("Robot")
    file_content.append("Romulan")
    file_content.append("Satyr")
    file_content.append("Seraphim")
    file_content.append("Shade")
    file_content.append("Shadow")
    file_content.append("Siren")
    file_content.append("Skeleton")
    file_content.append("Specter")
    file_content.append("Sphinx")
    file_content.append("Tauren")
    file_content.append("Tiefling")
    file_content.append("Time Lord")
    file_content.append("Troll")
    file_content.append("Undead")
    file_content.append("Valkyrie")
    file_content.append("Vampire")
    file_content.append("Voidelf")
    file_content.append("Vulcan")
    file_content.append("Werewolf")
    file_content.append("Will-o'-the-wisp")
    file_content.append("Wookiee")
    file_content.append("Worgen")
    file_content.append("Wraith")
    file_content.append("Xenomorph")
    file_content.append("Zandalari")
    file_content.append("Zombie")

    filepath = os.path.join(prompt_path, csv_file_name_4)
    cstr(f'Creating File `{filepath.strip()}`.').warning.print()
        
    with open(filepath, "w", encoding="utf-8", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write each line as a separate row in the CSV file
        for line in file_content:           
            try:
                csv_writer.writerow([line])    
            except:
                continue
        
    return ()
def create_file_5(prompt_path):
    file_content = []
    file_content.append("Camel")
    file_content.append("Cardinal")
    file_content.append("Cat")
    file_content.append("Cavalier King Charles Spaniel")
    file_content.append("Chameleon")
    file_content.append("Cheetah")
    file_content.append("Chickadee")
    file_content.append("Chicken")
    file_content.append("Chihuahua")
    file_content.append("Chimera")
    file_content.append("Chimpanzee")
    file_content.append("Chupacabra")
    file_content.append("Condor")
    file_content.append("Cornish Rex Cat")
    file_content.append("Cow")
    file_content.append("Coyote")
    file_content.append("Crocodile")
    file_content.append("Crow")
    file_content.append("Cuckoo")
    file_content.append("Dachshund")
    file_content.append("Deer")
    file_content.append("Devon Rex Cat")
    file_content.append("Diplodocus")
    file_content.append("Doberman Pinscher")
    file_content.append("Dolphin")
    file_content.append("Donkey")
    file_content.append("Duck")
    file_content.append("Egyptian Mau Cat")
    file_content.append("Elephant")
    file_content.append("Elk")
    file_content.append("English Mastiff")
    file_content.append("Exotic Shorthair Cat")
    file_content.append("Falcon")
    file_content.append("Ferret")
    file_content.append("Finch")
    file_content.append("Fish")
    file_content.append("Flamingo")
    file_content.append("Fox")
    file_content.append("French Bulldog")
    file_content.append("Frog")
    file_content.append("Gazelle")
    file_content.append("Gecko")
    file_content.append("German Shepherd")
    file_content.append("Giraffe")
    file_content.append("Goat")
    file_content.append("Golden Eagle")
    file_content.append("Golden Retriever")
    file_content.append("Goose")
    file_content.append("Gorilla")
    file_content.append("Great Dane")
    file_content.append("Griffin")
    file_content.append("Gyarados")
    file_content.append("Hare")
    file_content.append("Hawk")
    file_content.append("Hedgehog")
    file_content.append("Himalayan Cat")
    file_content.append("Horse")
    file_content.append("Hummingbird")
    file_content.append("Hyena")
    file_content.append("Jack Russell Terrier")
    file_content.append("Jaguar")
    file_content.append("Japanese Bobtail Cat")
    file_content.append("Jellyfish")
    file_content.append("Jigglypuff")
    file_content.append("Kangaroo")
    file_content.append("Kestrel")
    file_content.append("Koala")
    file_content.append("Korat Cat")
    file_content.append("Kraken")
    file_content.append("Labrador Retriever")
    file_content.append("Ladybird")
    file_content.append("Lark")
    file_content.append("Lemur")
    file_content.append("Leopard")
    file_content.append("Leviathan")
    file_content.append("Lion")
    file_content.append("Lizard")
    file_content.append("Llama")
    file_content.append("Lobster")
    file_content.append("Magpie")
    file_content.append("Maine Coon Cat")
    file_content.append("Mallard")
    file_content.append("Mammoth")
    file_content.append("Manx Cat")
    file_content.append("Mastodon")
    file_content.append("Megalodon")
    file_content.append("Mink")
    file_content.append("Moth")
    file_content.append("Mouse")
    file_content.append("Mule")
    file_content.append("Newt")
    file_content.append("Nightingale")
    file_content.append("Norwegian Forest Cat")
    file_content.append("Octopus")
    file_content.append("Opossum")
    file_content.append("Orangutan")
    file_content.append("Oriental Shorthair Cat")
    file_content.append("Osprey")
    file_content.append("Ostrich")
    file_content.append("Otter")
    file_content.append("Owl")
    file_content.append("Owlbear")
    file_content.append("Oyster")
    file_content.append("Panda")
    file_content.append("Panther")
    file_content.append("Parrot")
    file_content.append("Peacock")
    file_content.append("Pegasus")
    file_content.append("Pelican")
    file_content.append("Persian Cat")
    file_content.append("Phoenix")
    file_content.append("Pig")
    file_content.append("Pigeon")
    file_content.append("Platypus")
    file_content.append("Poodle")
    file_content.append("Porpoise")
    file_content.append("Puffin")
    file_content.append("Puppy Dog")
    file_content.append("Python")
    file_content.append("Rabbit")
    file_content.append("Raccoon")
    file_content.append("Ragdoll Cat")
    file_content.append("Rat")
    file_content.append("Raven")
    file_content.append("Red Panda")
    file_content.append("Rhinoceros")
    file_content.append("Robin")
    file_content.append("Rottweiler")
    file_content.append("Russian Blue Cat")
    file_content.append("Sabertooth tiger")
    file_content.append("Saint Bernard")
    file_content.append("Salamander")
    file_content.append("Scorpion")
    file_content.append("Scottish Fold Cat")
    file_content.append("Sea serpent")
    file_content.append("Seagull")
    file_content.append("Shar Pei")
    file_content.append("Shark")
    file_content.append("Sheep")
    file_content.append("Shih Tzu")
    file_content.append("Shrimp")
    file_content.append("Siamese Cat")
    file_content.append("Siberian Husky")
    file_content.append("Skunk")
    file_content.append("Sloth")
    file_content.append("Snail")
    file_content.append("Snake")
    file_content.append("Sparrow")
    file_content.append("Sphynx Cat")
    file_content.append("Spice worm")
    file_content.append("Spider")
    file_content.append("Squid")
    file_content.append("Squirrel")
    file_content.append("Starfish")
    file_content.append("Stork")
    file_content.append("Swan")
    file_content.append("Swordfish")
    file_content.append("Tapir")
    file_content.append("Tiger")
    file_content.append("Toad")
    file_content.append("Tortoise")
    file_content.append("Toucan")
    file_content.append("Turtle")
    file_content.append("Vulture")
    file_content.append("Wasp")
    file_content.append("Water buffalo")
    file_content.append("Weasel")
    file_content.append("Whale")
    file_content.append("Wolf")
    file_content.append("Wombat")
    file_content.append("Woodpecker")
    file_content.append("Woolly rhinoceros")
    file_content.append("Yak")
    file_content.append("Yorkshire Terrier")
    file_content.append("Zebra")

    filepath = os.path.join(prompt_path, csv_file_name_5)

    cstr(f'Creating File `{filepath.strip()}`.').warning.print()
        
    with open(filepath, "w", encoding="utf-8", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write each line as a separate row in the CSV file
        for line in file_content:           
            try:
                csv_writer.writerow([line])    
            except:
                continue
        
    return ()
def create_file_6(prompt_path):
    file_content = []
    file_content.append("2 wheeled vehicle")
    file_content.append("3 wheeled vehicle")
    file_content.append("4 wheeled vehicle")
    file_content.append("6 wheeled vehicle")
    file_content.append("Aircraft carrier")
    file_content.append("Airship")
    file_content.append("Ambulance")
    file_content.append("Armored vehicle")
    file_content.append("Aston Martin")
    file_content.append("Audi")
    file_content.append("Battlestar Galactica")
    file_content.append("Bicycle")
    file_content.append("BMW")
    file_content.append("Boat")
    file_content.append("Bodyboard")
    file_content.append("Bullet Train")
    file_content.append("Bus")
    file_content.append("Camper")
    file_content.append("Canoe")
    file_content.append("Car")
    file_content.append("Chair lift")
    file_content.append("Chevrolet")
    file_content.append("Container ship")
    file_content.append("Cruise ship")
    file_content.append("Ducati")
    file_content.append("Electric Car")
    file_content.append("Electric Vehicle")
    file_content.append("Ferrari")
    file_content.append("Ferrari 275 GTS 1966")
    file_content.append("Fire truck")
    file_content.append("Ford")
    file_content.append("Ford Mustang 1967 Shelby GT")
    file_content.append("Formula 1 car")
    file_content.append("Funicular")
    file_content.append("Garbage truck")
    file_content.append("Gondola")
    file_content.append("Hang glider")
    file_content.append("Harley-Davidson")
    file_content.append("Helicopter")
    file_content.append("High-speed Train")
    file_content.append("Honda")
    file_content.append("Horse-drawn carriage")
    file_content.append("Hot air balloon")
    file_content.append("Hovercraft")
    file_content.append("hypercar")
    file_content.append("Icebreaker")
    file_content.append("Jeep")
    file_content.append("Jet ski")
    file_content.append("Kawasaki")
    file_content.append("Kayak")
    file_content.append("Kiteboard")
    file_content.append("Lamborghini")
    file_content.append("Mega yacht")
    file_content.append("Mercedes-Benz")
    file_content.append("Motorcycle")
    file_content.append("Muscle car")
    file_content.append("Nissan")
    file_content.append("Nuclear submarine")
    file_content.append("Oil tanker")
    file_content.append("Paddle boat")
    file_content.append("Parachute")
    file_content.append("Plane")
    file_content.append("Police car")
    file_content.append("Porsche")
    file_content.append("Postal delivery truck")
    file_content.append("Raft")
    file_content.append("Rocketship")
    file_content.append("Roller skates")
    file_content.append("Rollerblades")
    file_content.append("Rope Bridge")
    file_content.append("Ropeway")
    file_content.append("Scooter")
    file_content.append("Seaplane carrier")
    file_content.append("Ship")
    file_content.append("Skateboard")
    file_content.append("Skis")
    file_content.append("Sled")
    file_content.append("Snow plow")
    file_content.append("Snowboard")
    file_content.append("Snowmobile")
    file_content.append("Space shuttle")
    file_content.append("Spacecraft")
    file_content.append("Spaceship")
    file_content.append("Sports car")
    file_content.append("Submarine")
    file_content.append("Surfboard")
    file_content.append("Tesla")
    file_content.append("Thunderbird")
    file_content.append("Tow truck")
    file_content.append("Toyota")
    file_content.append("Tractor trailer")
    file_content.append("Train")
    file_content.append("Tram")
    file_content.append("Trike")
    file_content.append("Trolley")
    file_content.append("Unicycle")
    file_content.append("Utility van")
    file_content.append("Volkswagen")
    file_content.append("Wakeboard")
    file_content.append("Water scooter")
    file_content.append("Water skis")
    file_content.append("Whaling ship")
    file_content.append("Wind surfboard")
    file_content.append("Yacht")
    file_content.append("Yamaha")


    filepath = os.path.join(prompt_path, csv_file_name_6)

    cstr(f'Creating File `{filepath.strip()}`.').warning.print()
        
    with open(filepath, "w", encoding="utf-8", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write each line as a separate row in the CSV file
        for line in file_content:           
            try:
                csv_writer.writerow([line])    
            except:
                continue
        
    return ()

NODE_NAME = 'Random Prompt by jice: Subjects [RvTools]'
NODE_DESC = 'Random Prompt (Subjects)'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvText_RandomPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
