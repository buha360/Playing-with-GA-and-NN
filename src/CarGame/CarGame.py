import math
import random
import pygame
import sys
import numpy as np

# Pygame inicializálása
pygame.init()

# Ablak beállításai
width, height = 1240, 768
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Neurális Hálózat Autós Játék')

# Színek
BLACK = (0, 0, 0)
PALYA_SZIN = (255, 255, 255)
AUTO_SZIN = (255, 0, 0)

# Autó beállításai
auto_meret = (20, 20)

checkpoint_vonal = [
    ((200, 721), (200, 643)),
    ((1000, 721), (1000, 643)),
    ((1153, 550), (1221, 550)),
    ((1153, 250), (1221, 250)),
    ((1050, 123), (1050, 186)),
    ((946, 300), (843, 300)),
    ((680, 413), (680, 496)),
    ((443, 300), (526, 300)),
    ((285, 113), (285, 186)),
    ((43, 550), (146, 550)),
    ((43, 250), (146, 250)),
]

palya_vonalak = [
    # Célegyenes
    ((150, 640), (1150, 640)),
    ((40, 725), (1225, 725)),

    # 2. egyenes
    ((1150, 640), (1150, 190)),
    ((1225, 725), (1225, 120)),

    # asd
    ((1225, 120), (840, 120)),
    ((1150, 190), (950, 190)),

    # asd
    ((840, 120), (840, 410)),
    ((950, 190), (950, 500)),

    # 3. egyenes
    ((840, 410), (530, 410)),
    ((950, 500), (440, 500)),

    # 3. kanyar
    ((530, 410), (530, 110)),
    ((440, 500), (440, 190)),

    # További szakaszok
    ((530, 110), (40, 110)),
    ((440, 190), (150, 190)),

    # Egyenes a célegyenes és a kezdővonal között
    ((150, 640), (150, 190)),
    ((40, 725), (40, 110)),
]


def vonalbol_kiterjesztett_teglalap(vonal, vastagsag=5):
    x1, y1 = vonal[0]
    x2, y2 = vonal[1]
    dx, dy = x2 - x1, y2 - y1
    hossz = ((dx ** 2) + (dy ** 2)) ** 0.5

    # Téglalap szélességének és magasságának meghatározása
    if abs(dx) > abs(dy):  # Vízszintes irányú vonal
        szel = hossz
        mag = vastagsag
        x = min(x1, x2)
        y = y1 - vastagsag / 2
    else:  # Függőleges irányú vonal
        szel = vastagsag
        mag = hossz
        x = x1 - vastagsag / 2
        y = min(y1, y2)

    return pygame.Rect(x, y, szel, mag)


class NeuralLayer:
    def __init__(self, input_count, neuron_count, weights=None, bias=None):
        if weights is None:
            self.weights = np.random.rand(input_count, neuron_count)
        else:
            self.weights = weights

        if bias is None:
            self.bias = np.random.rand(neuron_count)
        else:
            self.bias = bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        output = self.sigmoid(np.dot(inputs, self.weights) + self.bias)
        return output


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, inputs):
        inputs = np.array(inputs)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs.squeeze()  # squeeze() függvény, egydimenziós belső tömböket eltávolítása


class Auto:
    def __init__(self, neural_network=None, genetikai_kod=None, indulasi_irany=None):
        if genetikai_kod is None:
            # Biztosítsuk, hogy a genetikai kód hossza 42 legyen
            self.genetikai_kod = [random.uniform(-1, 1) for _ in range(42)]
        else:
            # Ha a megadott genetikai kód nem 42 hosszúságú, akkor hibaüzenetet adunk
            if len(genetikai_kod) != 42:
                raise ValueError("A genetikai kód hossza nem megfelelő (42 kell, hogy legyen)")
            self.genetikai_kod = genetikai_kod

        if neural_network is None:
            self.neural_network = self.genetikai_kod_alapjan_neuralis_halozat_letrehozasa()
        else:
            self.neural_network = neural_network

        if indulasi_irany is None:
            self.irany = random.uniform(0, 360)
        else:
            self.irany = indulasi_irany

        self.kezdeti_x, self.kezdeti_y = 500, 672
        self.x, self.y = self.kezdeti_x, self.kezdeti_y
        self.sebesseg = 0.8
        self.irany = random.uniform(0, 360)
        self.szenzorok_szama = 5  # Például 5 szenzor
        self.legutobbi_checkpointok = []
        self.el = True
        self.checkpointok = set()
        self.utolso_checkpoint = None
        self.eletben_toltott_ido = 0
        self.utolso_checkpoint_ido = 0

    def megtett_tavolsag(self):
        return math.sqrt((self.x - self.kezdeti_x) ** 2 + (self.y - self.kezdeti_y) ** 2)

    def checkpoint_ellenorzes(self, checkpoint_vonal):
        auto_rect = pygame.Rect(self.x, self.y, auto_meret[0], auto_meret[1])
        for vonal in checkpoint_vonal:
            checkpoint_rect = vonalbol_kiterjesztett_teglalap(vonal, 5)
            if auto_rect.colliderect(checkpoint_rect) and vonal != self.utolso_checkpoint:
                # Hozzáadjuk a checkpointot a legutóbbiak listájához
                self.legutobbi_checkpointok.append(vonal)
                if len(self.legutobbi_checkpointok) > 4:
                    # Csak az utolsó négy checkpointot tartjuk meg
                    self.legutobbi_checkpointok.pop(0)

    def eleg_diverz_checkpointok(self):
        # Ellenőrizzük, hogy az utolsó három checkpoint között van-e változatosság
        return len(set(self.legutobbi_checkpointok)) > 1

    def genetikai_kod_alapjan_neuralis_halozat_letrehozasa(self):
        layer_sizes = [5, 5, 2]
        layers = []
        start = 0
        for i in range(len(layer_sizes) - 1):
            input_count = layer_sizes[i]
            neuron_count = layer_sizes[i + 1]
            end = start + (input_count * neuron_count)
            weights = np.array(self.genetikai_kod[start:end]).reshape((input_count, neuron_count))

            # Ellenőrizze, hogy elegendő adat van-e a bias inicializálására
            bias_end = end + neuron_count
            if bias_end <= len(self.genetikai_kod):
                bias = np.array(self.genetikai_kod[end:bias_end])
            else:
                # Ha nincs elegendő adat, generáljon véletlenszerű értékeket
                bias = np.random.rand(neuron_count)

            layer = NeuralLayer(input_count, neuron_count, weights, bias)
            #print(f"Layer {i} - weights: {weights}, bias: {bias}")
            layers.append(layer)
            start = bias_end
        return NeuralNetwork(layers)

    def mozog(self):
        radian = math.radians(self.irany)
        self.x += math.cos(radian) * self.sebesseg
        self.y += math.sin(radian) * self.sebesseg
        #print(f"Auto.mozog - új pozíció: x={self.x}, y={self.y}, irány={self.irany}")

    def rajzol(self, screen):
        if self.el:
            # Autó középpontjának meghatározása
            auto_kozeppont_x = self.x + auto_meret[0] / 2
            auto_kozeppont_y = self.y + auto_meret[1] / 2

            # Autó rajzolása
            pygame.draw.rect(screen, AUTO_SZIN, pygame.Rect(self.x, self.y, 20, 20))

            # Frissített szenzor irányok
            szenzor_iranyok = [0, math.pi / 4, -math.pi / 4,  math.radians(75), -math.radians(75)]
            szenzor_hossza = 100
            for rel_irany in szenzor_iranyok:
                abszolut_irany = math.radians(self.irany) + rel_irany
                szenzor_vegpont_x = auto_kozeppont_x + math.cos(abszolut_irany) * szenzor_hossza
                szenzor_vegpont_y = auto_kozeppont_y + math.sin(abszolut_irany) * szenzor_hossza
                pygame.draw.line(screen, (255, 0, 0), (auto_kozeppont_x, auto_kozeppont_y),
                                 (szenzor_vegpont_x, szenzor_vegpont_y))

        # Checkpoint vonal rajzolása
        for vonal in checkpoint_vonal:
            pygame.draw.line(screen, (0, 255, 0), vonal[0], vonal[1], 5)

    def utkozes(self, teglalapok):
        if not self.el:
            return

        auto_rect = pygame.Rect(self.x, self.y, 20, 20)
        for teglalap in teglalapok:
            if auto_rect.colliderect(teglalap):
                #print(f"Ütközés! Autó pozíciója: x={self.x}, y={self.y}")
                self.el = False
                megtett_tavolsag = math.sqrt((self.x - self.kezdeti_x) ** 2 + (self.y - self.kezdeti_y) ** 2)
                #print(f"Autó meghalt. Megtett távolság: {megtett_tavolsag}")
                break

    def szenzorok_frissitese(self, palya_teglalapok):
        # Módosított szenzor irányok
        szenzor_iranyok = [0, math.pi / 4, -math.pi / 4, math.radians(75), -math.radians(75)]
        self.szenzor_ertekek = []

        for irany in szenzor_iranyok:
            szenzor_ertek = self.szenzor_ertek_szamitasa(irany, palya_teglalapok)
            self.szenzor_ertekek.append(szenzor_ertek)

    def szenzor_ertek_szamitasa(self, rel_irany, palya_teglalapok):
        # A sugarak hossza és iránya
        szenzor_hossza = 100  # 100 pixel

        # Az autó aktuális irányához hozzáadjuk a relatív irányt
        abszolut_irany = math.radians(self.irany) + rel_irany

        # Lépésenként vizsgálja a sugár útvonalát
        for i in range(1, szenzor_hossza + 1):
            ellenorzes_x = self.x + math.cos(abszolut_irany) * i
            ellenorzes_y = self.y + math.sin(abszolut_irany) * i

            # Ellenőrizze az ütközést minden pálya téglalappal
            for teglalap in palya_teglalapok:
                if teglalap.collidepoint(ellenorzes_x, ellenorzes_y):
                    # Ha ütközés történt, visszaadja a távolságot
                    return math.sqrt((ellenorzes_x - self.x) ** 2 + (ellenorzes_y - self.y) ** 2)

        # Ha nincs ütközés, a teljes szenzor hosszát adja vissza
        return szenzor_hossza

    def dontes(self, neural_output):
        if len(neural_output) >= 2:
            irany_valtas = (neural_output[0] - 0.5) * 90  # Irányváltás -45 és +45 fok között
            sebesseg_valtozas = neural_output[1] * 3  # Sebesség változás 0 és 3 között
            self.irany = (self.irany + irany_valtas) % 360
            self.sebesseg = max(0.1, min(1, self.sebesseg + sebesseg_valtozas))
        else:
            print("Hiba: A neurális hálózat kimenete nem megfelelő méretű.")

    def update(self):
        self.szenzorok_frissitese(palya_teglalapok)
        neural_input = self.szenzor_ertekek
        neural_output = self.neural_network.predict(neural_input)
        self.dontes(neural_output)
        self.mozog()
        self.utkozes(palya_teglalapok)
        self.rajzol(screen)

        if self.el:
            self.eletben_toltott_ido += 1
            if self.eletben_toltott_ido - self.utolso_checkpoint_ido > 20:  # 10 másodperces cooldown
                self.checkpoint_ellenorzes(checkpoint_vonal)
                self.utolso_checkpoint_ido = self.eletben_toltott_ido

    def print_genetikai_kod(self):
        print("Genetikai kód:", self.genetikai_kod)


# Értékelési funkció
def ertekel(auto):
    checkpoint_pontok = len(auto.checkpointok) * 100
    return checkpoint_pontok


# Kiválasztási funkció
def kivalaszt(populacio):
    # Minden autó értékelése
    ertekelt_populacio = [(auto, ertekel(auto)) for auto in populacio]
    # Pontszám alapján sorba rendezés
    ertekelt_populacio.sort(key=lambda x: x[1], reverse=True)

    valogatott = []
    for auto, pontszam in ertekelt_populacio:
        # Ellenőrizzük, hogy az autó checkpoint viselkedése megfelel-e a kritériumoknak
        if auto.eleg_diverz_checkpointok():
            # Csak azokat az autókat választjuk ki, amelyek megfelelnek az új kritériumnak
            valogatott.append((auto, pontszam))

    # A válogatott autók egy részének visszaadása
    return valogatott[:len(populacio) // 2]


# Keresztezési funkció
def keresztez(szulo1, szulo2):
    uj_genetikai_kod = []
    genetikai_kod_hossza = 42  # Biztosítsuk, hogy a genetikai kód hossza 42
    for i in range(genetikai_kod_hossza):
        if random.random() < 0.5:
            uj_genetikai_kod.append(szulo1.genetikai_kod[i])
        else:
            uj_genetikai_kod.append(szulo2.genetikai_kod[i])
    return Auto(genetikai_kod=uj_genetikai_kod)


# Mutációs funkció
def mutacio(auto):
    mutacios_esely = 0.05  # 5% esély a mutációra
    mutacios_mertek = 0.02  # A mutáció mértéke: 100%-ból csak 2% genetikai kód mutálódik
    for i in range(len(auto.genetikai_kod)):
        if random.random() < mutacios_esely:
            # Mutáció alkalmazása és érték korlátozása -1 és 1 között
            uj_ertek = auto.genetikai_kod[i] + random.uniform(-mutacios_mertek, mutacios_mertek)
            auto.genetikai_kod[i] = max(-1, min(1, uj_ertek))


def generacio_valtas(populacio):
    # Kiválasztás, keresztezés és mutáció

    print("\n--- Genetikai kódok az aktuális generációban ---")
    for auto in populacio:
        auto.print_genetikai_kod()

    legjobbak = kivalaszt(populacio)
    uj_populacio = []
    while len(uj_populacio) < len(populacio):
        szulo1 = random.choice(legjobbak)[0]
        szulo2 = random.choice(legjobbak)[0]
        gyermek = keresztez(szulo1, szulo2)
        mutacio(gyermek)
        uj_populacio.append(gyermek)

    populacio[:] = uj_populacio
    # Nincs szükség a genetikai kód újbóli véletlenszerű beállítására


# Generációk száma
generaciok_szama = 100

# Kezdeti populáció létrehozása
populacio = [Auto() for _ in range(50)]

palya_teglalapok = [vonalbol_kiterjesztett_teglalap(vonal) for vonal in palya_vonalak]

# Játékciklus
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Generációváltás manuális indítása
                generacio_valtas(populacio)

    screen.fill(BLACK)
    # Pálya kirajzolása
    for vonal in palya_vonalak:
        pygame.draw.line(screen, PALYA_SZIN, vonal[0], vonal[1], 5)
    for teglalap in palya_teglalapok:
        pygame.draw.rect(screen, PALYA_SZIN, teglalap)

    mindenki_meghalt = True
    # Autók frissítése és kirajzolása
    for auto in populacio:
        if auto.el:
            mindenki_meghalt = False
        auto.update()
        auto.rajzol(screen)

    # Automatikus generációváltás, ha mindenki meghalt
    if mindenki_meghalt:
        generacio_valtas(populacio)

    pygame.display.flip()  # Képernyő frissítése

# Kilépés pygame-ból
pygame.quit()
sys.exit()
