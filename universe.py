# Universe module - where most all game logic resides

from nutil import *
from nutil import lists
import numpy as np
import math
import collections
import copy
import functools
import json

import common
import components
import agency

TITLE = 'Dark Universe'
VERSION = 0.007
FULL_TITLE = f'{TITLE} v{VERSION}'


PLANK = PLANK_LENGTH = 10**-5
DIMENSIONS = 2
PROXIMITY_DISTANCE = 10**-1
np.random.seed(Seed().randint(0, 999999))

# Data directory
ROOT_DIR = give_usr_dir('darku')
SAVE_DIR = ROOT_DIR / 'saves'
if not SAVE_DIR.is_dir():
    SAVE_DIR.mkdir()
LOG_DIR = ROOT_DIR / 'logs'
if not LOG_DIR.is_dir():
    LOG_DIR.mkdir()
LOG_PATH = LOG_DIR / 'log.txt'
log_disk(LOG_PATH, make_title('Debug log'), timestamp=True, clear=True)


TINY_DEV_MODE = False
# TINY_DEV_MODE = True

# DEFAULT_SEED = 'dev'
DEFAULT_SEED = Seed().r
GEN_FRACTAL = [
    (2, 2),
    (7, 12),
    (8, 13),
    (-2, 6),
    (-2, 1),
    ]
UID_TYPES = ['body', 'bridge', 'player', 'universe']
UID_ILLEGAL_STRS = ['¦¦']
PLAYER_PID = '╣Player╠'
ILLEGAL_UIDS = [PLAYER_PID]

DEFAULT_ROCK_SUBCLASSES = {
    '0': {'name': '■ SuperMassive Body',
          'description': 'A hugely massive giant',
          'spawn_bias': 0.01, 'spawn_capacity': 20,
          'survey_cost': 10,
          'color': (0, 1, 0.5)},
    '1': {'name': '╬ Pulsar',
          'description': 'Massive spinning giant',
          'spawn_bias': 0.05, 'spawn_capacity': 2,
          'survey_cost': 50,
          'color': (0, 1, 0.5)},
    '2': {'name': '§ Planet',
          'description': 'Massive rock',
          'spawn_bias': 0.1, 'spawn_capacity': 1,
          'survey_cost': 100,
          'color': (0.2, 1, 0.5)},
    '3': {'name': 'ø Moon',
          'description': 'Large rock',
          'spawn_bias': 0.2, 'spawn_capacity': .5,
          'survey_cost': 200,
          'color': (0.4, 1, 0.5)},
    '4': {'name': '¤ Roid',
          'description': 'Low density rock',
          'spawn_bias': 0.4, 'spawn_capacity': .2,
          'survey_cost': 500,
          'color': (0.6, 1, 0.5)},
    '5': {'name': '• Roidlet',
          'description': 'High density rock',
          'spawn_bias': 3.0, 'spawn_capacity': .1,
          'survey_cost': 1000,
          'color': (0.8, 1, 0.5)},
    }
SUBCLASS_COLORS = {f'{i}': (0.2*i, 1, 0.6) for i in range(6)}

if TINY_DEV_MODE:
    GEN_FRACTAL = [(3, 3) for _ in range(len(GEN_FRACTAL))]


def check_uid_legal(uid):
    if uid in ILLEGAL_UIDS:
        return False
    for illegal in UID_ILLEGAL_STRS:
        if illegal in uid:
            return False
    return True


class Universe:
    def __init__(self, time=None, game_over=None,
                 seed=None, seed_gen=None, seed_rocks=None,
                 seed_players=None, seed_colors=None,
                 simsched=None, sim_halt_flag=None, sim_stats=None,
                 rock_subclasses=None, moria=None,
                 moria_offset=None, moria_fractal_index=None,
                 fractal_children_counts=None, subclasses=None,
                 buid_count=None, cuid_count=None,
                 pid_count=None, do_genesis=True):
        self.__game_over = False if game_over is None else game_over
        self.time = 0 if time is None else time
        self.debug(make_title('Creating Universe', length=50, nonewline=True), force_print=True)
        self.seed = Seed(DEFAULT_SEED) if seed is None else seed
        self.debug(f'seed: {self.seed}', force_print=True)
        self.seed_gen = Seed(self.seed.r) if seed_gen is None else seed_gen
        self.seed_rocks = Seed(self.seed.r) if seed_rocks is None else seed_rocks
        self.seed_players = Seed(self.seed.r) if seed_players is None else seed_players
        self.seed_colors = Seed(self.seed.r) if seed_colors is None else seed_colors
        if seed_colors is None:
            self.seed_colors = Seed(seed_colors)
        self.seed_colors = Seed('colors2')
        # Simulation
        self.simulation_scheduler = Scheduler() if simsched is None else simsched
        self._simulation_halt_flag = False if sim_halt_flag is None else sim_halt_flag
        self.__simrate = RateCounter(sample_size=25)
        self.sim_stats = {'total_events_processed': 0} if sim_stats is None else sim_stats
        # Universe
        self._buid_count = 0 if buid_count is None else buid_count
        self.bodies = {}
        # Rocks
        self.rock_subclasses = self.default_rock_subclasses if rock_subclasses is None else rock_subclasses
        self.moria = np.zeros(2) if moria is None else moria
        self.moria_offset = np.zeros(2) if moria_offset is None else moria_offset
        self._moria_fractal_index = 3 if moria_fractal_index is None else moria_fractal_index
        self.fractal_children_counts = GEN_FRACTAL if fractal_children_counts is None else fractal_children_counts
        self.genesis_object = common.Genesis(self.seed_gen, fractal_children=self.fractal_children_counts)
        # Ships
        self._cuid_count = 0 if cuid_count is None else cuid_count
        self.subclasses = self.default_subclasses if subclasses is None else subclasses
        # Players
        self._pid_count = 0 if pid_count is None else pid_count
        self.players = {}
        self._player_pid = PLAYER_PID

        # Genesis
        if do_genesis:
            self.genesis()
            self.simulate_time(1)
            # self.interval_event(context='')

        # Misc
        self.debug(make_title('Universe Created', length=50, nonewline=True), force_print=True)

    def do_export(self, savename='test'):
        exported_uni = json.dumps(self.__dict__, cls=DUEncoder)
        pretty_str = adis(json.loads(exported_uni), key_cap=40, value_cap=100)
        # del exported_uni['genesis_object']
        file = SAVE_DIR / f'{savename}.dus'
        prettyfile = SAVE_DIR / f'export.pretty'
        # Save export data to disk
        file_dump(file, exported_uni)
        file_dump(prettyfile, pretty_str)
        self.debug(f'Saved game to file: {file}', force_print=True)

    @classmethod
    def do_import(cls, savename='test'):
        file = SAVE_DIR / f'{savename}.dus'
        lf = file_load(file)
        d = json.loads(lf)
        print(sjoin([
            f'Importing universe from {len(lf):,} bytes of data.',
            f'Imported seed: {d["seed"]}',
            f'Imported time: {d["time"]}',
            ]))
        uni = cls(
            time=d['time'],
            seed=Seed(*d['seed']),
            seed_gen=Seed(*d['seed_gen']),
            seed_colors=Seed(*d['seed_colors']),
            simsched=Scheduler.import_decode(d['simulation_scheduler']),
            sim_halt_flag=d['_simulation_halt_flag'],
            sim_stats=d['sim_stats'],
            rock_subclasses=d['rock_subclasses'],
            moria=d['moria'],
            moria_offset=np.asarray(d['moria_offset']),
            moria_fractal_index=d['_moria_fractal_index'],
            fractal_children_counts=d['fractal_children_counts'],
            subclasses=d['subclasses'],
            buid_count=d['_buid_count'],
            cuid_count=d['_cuid_count'],
            pid_count=d['_pid_count'],
            do_genesis=False,
            )
        # Since both bodies and players expect to be passed the universe object itself on init, we generate them post-hoc
        # Import each player (starting with players since ships refer to them on init)
        for pid, pd in d['players'].items():
            uni.players[pid] = agency.Player.import_decode(uni, pd)
        # Import each body
        for buid, bd in d['bodies'].items():
            # First we must recognize if this body is a rock or ship
            if 'bridge' in bd:
                uni.bodies[buid] = Ship.import_decode(uni, bd)
            if 'elements' in bd:
                uni.bodies[buid] = Rock.import_decode(uni, bd)
        return uni

    def debug(self, message, force_print=False):
        message = f'{disn(self.time, sig_digits=10, precision=6)} ¦ {message}'
        log_disk(LOG_PATH, message, force_print=force_print)

    def game_over(self):
        self.__game_over = True
        self.request_simulation_halt()

    # SUBCLASSES
    def subclass_color(self, subclass):
        s = Seed(f'{self.seed_colors}-{subclass}')
        return f'#{"".join(decimal2hex(s.randfloat()) for i in range(3))}'

    def add_subclass(self, name, description, stats):
        cuid = self.get_new_cuid()
        self.subclasses[cuid] = {
            'name': name,
            'description': description,
            'stats': stats,
            }
        return cuid

    def name2cuid(self, name):
        for cuid, subclass in self.subclasses.items():
            if name in subclass['name']:
                return cuid
        return None

    @property
    def default_rock_subclasses(self):
        return DEFAULT_ROCK_SUBCLASSES

    @property
    def default_subclasses(self):
        s = {}
        return s

    def subclass2color(self, subclass):
        if subclass in self.subclasses:
            sc = self.subclasses[subclass]
        elif subclass in self.rock_subclasses:
            sc = self.rock_subclasses[subclass]
        if 'color' in sc:
            return sc['color']
        if subclass in SUBCLASS_COLORS:
            return SUBCLASS_COLORS[subclass]
        s = Seed(subclass)
        return (s.r, 1, 1)

    # GENESIS
    def genesis(self):
        def get_fractal_name(index):
            return Seed(f'{self.seed_gen}-{index}').choice(lists.CELESTIAL_NAMES)

        # Set Moria location
        spawnables = [*filter(lambda x: len(x[0])==self._moria_fractal_index, self.genesis_object.nodes)]
        for potential_spawn in self.genesis_object.nodes:
            if len(potential_spawn[0]) == self._moria_fractal_index:
                spawn_index, spawn_location = potential_spawn
                break
        else:
            raise RuntimeError(f'Found no spawnable body. Check genesis fractal configuration.')
        # self.moria_offset = spawn
        self.moria = spawn_location

        # Systems
        self.ancestry = {}
        for node in self.genesis_object.nodes:
            index, location = node
            findex = len(index)
            cuid = str(findex)
            sc = self.rock_subclasses[cuid]
            mass = 10**self.seed_rocks.randfloat(0, 2) * common.Genesis.SCALES[cuid]['mass']
            name = f'{get_fractal_name(index)}'
            if findex == 3:
                name = f'{get_fractal_name(index)}'
            elif findex == 4:
                name = f'{get_fractal_name(index[:-1])}.{index[-1]+1}'
            elif findex == 5:
                name = f'{get_fractal_name(index[:-2])}.{index[-2]+1}{lists.LETTERS[index[-1]]}'
            name = f'{common.Names.rock_prefixes[findex]} {name}'
            elements = None
            if index is spawn_index:
                elements = {}
                for spawn_ei in (0, 2, 4, 6):
                    cap = sc['spawn_capacity'] / (2**(spawn_ei/2)) / 2
                    elements[common.E.NAMES[spawn_ei]] = {
                        'capacity': cap,
                        'real_quantity': cap * Rock.MAX_REAL_QUANTITY_FACTOR,
                        'accessibility': 0.2-(spawn_ei*0.025),
                        }
            new_buid = self.new_rock(
                name=name,
                subclass=cuid,
                ancestry=index,
                mass=mass,
                seed=self.seed_rocks.r,
                starting_position=location,
                elements=elements)
            if index is spawn_index:
                spawn_buid = new_buid
            self.ancestry[str(index)] = new_buid

        self.player_genesis(spawn_buid)

        self.debug(f'Total objects: {len(self.bodies)}', force_print=True)

    def player_genesis(self, spawn_buid):
        # Starting player ships
        starting_ships = [
            self.add_subclass(**sc) for sc in components.STARTING_SHIPS_STATS]
        # Player
        self.add_player(agency.Player(
            uni=self,
            pid=self._player_pid,
            seed=Seed(self.seed_players.r),
            name='Dev',
            spawn_buid=spawn_buid,
            ))
        self.player.generate_starting_ships(starting_ships)

    # SIMULATION
    def simevent_in(self, t):
        """Checks if there is a timed callback before target time"""
        if len(self.simulation_scheduler) > 0:
            if self.simulation_scheduler.entries[0][0] <= t:
                return True
        return False

    def simulate_time(self, elapsed, allow_log=False, max_blocking=2000):
        """Executes timed callbacks until target time is reached."""
        if self.__game_over:
            self.debug('GAME OVER :(', force_print=True)
            # return
        self._simulation_halt_flag = False
        initial_time = self.time
        target_time = self.time + elapsed

        start_time = ping()
        max_block_concede = False
        while self.simevent_in(target_time) and not self._simulation_halt_flag and not max_block_concede:
            self.__simrate.ping()
            self.next_callback()
            callback_run_time = self.__simrate.pong()
            if pong(start_time) > max_blocking:
                self.player.alert(f'Simulation exceeded blocking time ({max_blocking/1000} seconds). Simulated: {round(self.time - initial_time, 3)} time.')
                max_block_concede = True

        if not self._simulation_halt_flag and not max_block_concede:
            self.__set_time(target_time)

        if allow_log:
            self.debug(make_title(f'Simulation time: {self.time}'))

    @property
    def simrate(self):
        return self.__simrate

    def __set_time(self, t):
        assert t >= self.time
        self.time = t

    def request_simulation_halt(self):
        """Called to halt simulation (simulate_time)."""
        self._simulation_halt_flag = True
        self._last_simulation_halt = self.time

    def add_timed_callback(self, time, otype, uid, callback, context):
        """
        Add a timed callback (with context) to the simulation scheduler.
        The time determines when this event should take place.
        The otype determines which type of uid is given.
        The uid is used to find the specific instance.
        The callback is the name of the method upon the given instance to call.
        The context is an arbirary dictionary to be passed as a parameter.
        """
        if time < self.time:
            raise AssertionError('Timed callback time must be greater than current time!')
        assert otype in UID_TYPES
        self.simulation_scheduler.add_event_sorted([time, otype, uid, callback, context])

    def next_callback(self):
        """Advances time to next event in simsched and executes it."""
        time, otype, uid, callback_name, context = self.simulation_scheduler.pop_next()
        assert time >= self.time
        assert otype in UID_TYPES
        if otype == 'body':
            obj = self.id2body(uid)
        elif otype == 'bridge':
            obj = self.id2body(uid).bridge
        elif otype == 'player':
            obj = self.id2player(uid)
        elif otype == 'universe':
            obj = self
        else:
            raise RuntimeError(f'Cannot find object type {otype} for simsched event callback.')
        callback = getattr(obj, callback_name)
        assert callable(callback)
        self.__set_time(time)
        # p = ping()
        callback(context)
        # print(sjoin([
        #     make_title(f'{otype} - {callback_name}',
        #                pre_line_space=True, nonewline=True),
        #     adis(context),
        #     f'Elapsed in: {pong(p)}ms',
        #     ]))
        self.sim_stats['total_events_processed'] += 1

    def interval_event(self, context):
        interval = 1000
        context = {'summary': f'Universal Interval @ {self.time + interval}'}
        self.add_timed_callback(self.time + interval, 'universe', None, 'interval_event', context)

    @property
    def proximity_distance(self):
        return PROXIMITY_DISTANCE

    # BODIES
    def check_velocity_interactions(self, buid):
        """This method is called when buid changes vv.
        This calls every other relevant body to make arbitrary checks regarding this new vv."""
        b = self.id2body(buid)
        # Iterate over each body in sector
        for target_buid, target_body in self.bodies.items():
            if buid == target_buid:
                continue
            # We allow buid and target_buid to check each other arbitrarily
            b.check_velocity_interactions(target_buid)
            target_body.check_velocity_interactions(buid)

    def get_body_distance(self, a, b):
        return np.linalg.norm(self.bodies[a].position - self.bodies[b].position)

    def get_new_buid(self):
        buid = Seed.h256(str(self._buid_count))[:20]
        while buid in self.bodies or not check_uid_legal(buid):
            self._buid_count += 1
            buid = Seed.h256(str(self._buid_count))[:20]
        return buid

    def get_new_cuid(self):
        cuid = Seed.h256(str(self._cuid_count))[:20]
        while cuid in self.subclasses:
            self._cuid_count += 1
            cuid = Seed.h256(str(self._cuid_count))[:20]
        return cuid

    def get_new_pid(self):
        pid = Seed.h256(str(self._pid_count))[:20]
        self._pid_count += 1
        while pid in self.players or not check_uid_legal(pid):
            pid = Seed.h256(str(self._pid_count))[:20]
            self._pid_count += 1
        return pid

    def new_rock(self, **kwargs):
        buid = self.get_new_buid()
        self._add_body(Rock(uni=self, buid=buid, **kwargs))
        return buid

    def new_ship(self, **kwargs):
        buid = self.get_new_buid()
        self._add_body(Ship(uni=self, buid=buid, **kwargs))
        return buid

    def _add_body(self, body):
        assert body.buid not in self.bodies
        # Iterate over all ships to notify of new body in sector
        self.bodies[body.buid] = body
        for existing_body in self.bodies.values():
            existing_body.new_body(body.buid)
            body.new_body(existing_body.buid)

    def _kill_body(self, buid):
        b = self.id2body(buid)
        assert b.is_ship
        for existing_body in self.bodies.values():
            existing_body.old_body(buid)

    @property
    def body_count(self):
        return len(self.bodies)

    def filter_bodies(self, bodies, **filters):
        return filter(lambda x, kw=filters: self.filter_body(x, **kw), bodies)

    def filter_body(self, buid, exists=True, allegiance=None,
                    rocks=True, ships=True, subclasses=None,
                    names=None, tags=None, case_sensitive=False,
                    ):
        """Check if a body passes filters. The names filter is exclusive - one of the name strings must exist in body name. Subclasses must be either None (to disable subclass filtering) or a list of subclasses"""
        body = self.id2body(buid)
        if exists and not body.exists:
            return False
        if names is not None:
            if case_sensitive:
                if not any(name in body.fname for name in names):
                    return False
            else:
                if not any((name.lower() in body.fname.lower() for name in names)):
                    return False
        if body.is_ship and tags is not None:
            if isinstance(tags, str):
                tags = tags,
            if not any(tag in body.tags for tag in tags):
                return False
        if (not rocks and body.is_rock) or (not ships and body.is_ship):
            return False
        if isinstance(subclasses, list) or isinstance(subclasses, tuple):
            if case_sensitive:
                if not any(ctext in body.resolve_subclass['name'] or ctext in body.resolve_subclass['description'] for ctext in subclasses):
                    return False
            else:
                if not any(ctext.lower() in body.resolve_subclass['name'].lower() or ctext.lower() in body.resolve_subclass['description'].lower() for ctext in subclasses):
                    return False
        if allegiance is not None:
            if body.is_rock:
                return False
            if body.is_ship:
                if body.allegiance != allegiance:
                    return False
        return True

    def gen_bodies(self, **kwargs):
        return filter(lambda x, kw=kwargs: self.filter_body(x.buid, **kw), self.bodies.values())

    def gen_buids(self, **kwargs):
        return (b.buid for b in self.gen_bodies(**kwargs))

    def id2bodies(self, buids):
        return [self.id2body(buid) for buid in buids]

    def id2body(self, buid):
        if buid not in self.bodies:
            self.debug(f'Failed to find buid {buid}.', force_print=True)
            raise IndexError(f'Failed to find buid {buid}')
        return self.bodies[buid]

    @property
    def buids(self):
        return list(self.bodies.keys())

    def print_bodies(self, bodies):
        print(adis(sjoin([self.id2body(b).fname for b in bodies])))

    # PLAYER
    @property
    def player(self):
        return self.players[self._player_pid]

    def add_player(self, player):
        self.players[player.pid] = player

    def id2player(self, pindex):
        return self.players[pindex]

    # UTILITY
    def when_at_range(self, delta, b1, b2):
        """
        == wolframalpha ==
        input: y = sqrt((a + t*c - e - t*g)^2 + (b + t*d - f - t*h)^2) ; solve for t
        output:
        t = (sqrt((2 a c - 2 a g + 2 b d - 2 b h - 2 c e - 2 d f + 2 e g + 2 f h)^2 - 4 (c^2 - 2 c g + d^2 - 2 d h + g^2 + h^2) (a^2 - 2 a e + b^2 - 2 b f + e^2 + f^2 - y^2)) - 2 a c + 2 a g - 2 b d + 2 b h + 2 c e + 2 d f - 2 e g - 2 f h)/(2 (c^2 - 2 c g + d^2 - 2 d h + g^2 + h^2))
        =======
        """
        a, b, c, d = self.bodies[b1].quadeq_parameters()
        e, f, g, h = self.bodies[b2].quadeq_parameters()
        try:
            den = 2*(c**2 - 2*c*g + d**2 - 2*d*h + g**2 + h**2)
            assert den != 0
            p1 = math.sqrt((2*a*c - 2*a*g + 2*b*d - 2*b*h - 2*c*e - 2*d*f + 2*e*g + 2*f*h)**2 - 4*(c**2 - 2*c*g + d**2 - 2*d*h + g**2 + h**2) * (a**2 - 2*a*e + b**2 - 2*b*f + e**2 + f**2 - delta**2))
            p2 =  - 2*a*c + 2*a*g - 2*b*d + 2*b*h + 2*c*e + 2*d*f - 2*e*g - 2*f*h
            t1 = (-p1 + p2)/den
            t2 = (p1 + p2)/den
            return t1, t2
        except:
            return None

    def sort_by_distance_key(self, buid, target):
        return vmag(self.id2body(buid).position - target)

    def sort_by_mass_key(self, buid):
        return -self.id2body(buid).dry_mass

    def sort_by_subclass_key(self, buid):
        return self.id2body(buid).subclass

    @property
    def PLANK(self):
        return PLANK

    @property
    def sim_halted(self):
        return self._simulation_halt_flag


class Body:
    def __init__(self, uni, buid, subclass, name=None, exists=True,
                 starting_position=None, starting_velocity=None, vv_context=None,
                 nearest_bodies=None, nearest_bodies_sorted=None, nearest_bodies_updated=None):
        self._subclass = subclass
        self._exists = exists
        self.uni = uni
        self.buid = buid
        self.name = f'Unnamed: {buid[:5]}' if name is None else name
        self.__anchor_position = np.random.random(2)*1000 if starting_position is None else np.asarray(starting_position)
        self.__anchor_time = self.time
        self.__velocity = np.zeros(2) if starting_velocity is None else np.asarray(starting_velocity)
        self.__vv_context = '' if vv_context is None else vv_context
        self.__nearest_bodies = {} if nearest_bodies is None else nearest_bodies
        self.__nearest_bodies_sorted = [] if nearest_bodies_sorted is None else nearest_bodies_sorted
        self.__nearest_bodies_updated = -1 if nearest_bodies_updated is None else nearest_bodies_updated

    def export_encode(self):
        e = {**self.__dict__}
        del e['uni']
        # Nearest bodies database inflates save file size considerably without much benefit
        del e['_Body__nearest_bodies']
        del e['_Body__nearest_bodies_sorted']
        del e['_Body__nearest_bodies_updated']
        return e

    def terminate_existence(self):
        self._exists = False
        self.uni._kill_body(self.buid)

    def rename(self, new_name):
        self.name = new_name

    @property
    def exists(self):
        return self._exists

    @property
    def fname(self):
        return f'{self.buid[:3]}.{self.buid[-3:]}| {self.name}'

    def __str__(self):
        return f'<B# {self.buid:0>5}|{self.name}>'

    @property
    def time(self):
        return self.uni.time

    @property
    def position(self):
        return self.get_position()

    @property
    def velocity(self):
        return vmag(self.__velocity)

    @property
    def vv(self):
        return self.__velocity

    def get_position(self, t=None):
        t = self.time if t is None else t
        return self.__anchor_position + self.__velocity * (t - self.__anchor_time)

    def get_line_points(self):
        return np.array(self.last_position), np.array(self.last_position)+np.array(self.velocity)

    def reapply_vv(self):
        self.set_vv(self.__velocity)

    def refresh_anchor(self):
        self.__anchor_position = self.position
        self.__anchor_time = self.time

    def set_vv(self, vv=None):
        if vv is None:
            vv = np.zeros(2)
        self.refresh_anchor()
        self.__velocity = np.array(vv)
        self.__vv_context = self.uni.seed.h256(f'{self.__anchor_position}{self.__anchor_time}{self.__velocity}')
        self.uni.check_velocity_interactions(self.buid)

    @staticmethod
    def get_random_velocity(maximum=1):
        return np.random.random(2)*maximum-(maximum/2)

    @property
    def in_motion(self):
        return vmag(self.vv) > PLANK_LENGTH

    def distance_to(self, buid):
        return vmag(self.position - self.uni.id2body(buid).position)

    @property
    def nearest_bodies(self):
        # Update the cache if needed
        if self.__nearest_bodies_updated < self.time:
            self.__nearest_bodies = {
                b: self.distance_to(b) for b in self.uni.bodies.keys() if b != self.buid
            }
            self.__nearest_bodies_sorted = list(sorted(self.__nearest_bodies.items(), key=lambda x: x[1]))
            self.__nearest_bodies_updated = self.time
        return self.__nearest_bodies_sorted

    def nearest_body(self, rocks=True, ships=True, subclass=None):
        for buid, distance in self.nearest_bodies:
            b = self.uni.id2body(buid)
            if (not rocks and b.is_rock) or (not ships and b.is_ship):
                continue
            if isinstance(subclass, list):
                if b.subclass in subclass:
                    return buid
            elif isinstance(subclass, int):
                if b.subclass == subclass:
                    return buid
            elif subclass is None:
                return buid
        return None

    def proximal(self, include_self=False, **kwargs):
        prox = [self.buid] if include_self else []
        for buid, distance in self.nearest_bodies:
            if distance > self.uni.proximity_distance:
                break
            if self.uni.filter_body(buid, **kwargs):
                prox.append(buid)
        return prox

    def quadeq_parameters(self):
        x, y = self.position
        dx, dy = self.vv
        return x, y, dx, dy

    def new_body(self, buid):
        pass

    def old_body(self, buid):
        pass

    @property
    def has_sensor(self):
        return False

    def check_velocity_interactions(self, target_buid):
        pass

    @property
    def vvcon(self):
        return self.__vv_context

    @property
    def subclass(self):
        return self._subclass

    @property
    def is_rock(self):
        return False

    @property
    def is_ship(self):
        return False

    @property
    def is_ally(self):
        return False

    @property
    def resolve_subclass(self):
        if self.is_rock:
            return self.uni.rock_subclasses[self._subclass]
        if self.is_ship:
            return self.uni.subclasses[self._subclass]
        raise TypeError(f'Base Body has no subclass.')


class Rock(Body):
    MAX_REAL_QUANTITY_FACTOR = 10000
    ACCESSIBILITY_FACTOR = 6
    ACCESSIBILITY_AVG_FACTOR = 3

    def __init__(self, mass, ancestry, seed,
                 elements=None, elements_anchor=None, **kwargs):
        super().__init__(**kwargs)
        self._ancestry = ancestry
        self.mass = mass
        self.seed = Seed(seed)
        self._elements_anchor = 0 if elements_anchor is None else elements_anchor
        self.elements = collections.defaultdict(lambda: {
            'capacity': 0,
            'real_quantity': 0,
            'accessibility': 0,
            })

        # Generate elements from seed or take from args
        if elements is None:
            # Spawn element capacity
            sample_size = max(1, self.seed.randint(-3, 7))
            spawned_elements = common.E.spawn_elements(
                seed=self.seed.r,
                bias=self.uni.rock_subclasses[self._subclass]['spawn_bias'],
                sample_size=sample_size)
            counts = collections.defaultdict(lambda: 0)
            spawn_cap = self.resolve_subclass['spawn_capacity']
            for eindex in spawned_elements:
                ename = common.E.i2e(eindex)
                counts[ename] += 1
                variation = self.seed.randfloat(0.3, 3)
                self.elements[ename]['capacity'] += spawn_cap * variation / sample_size

            # Determine accessibility and real quantity for each element
            for ename, element in self.elements.items():
                access = sum([self.seed.r**Rock.ACCESSIBILITY_FACTOR for _ in range(Rock.ACCESSIBILITY_AVG_FACTOR)]) / Rock.ACCESSIBILITY_AVG_FACTOR
                element['accessibility'] = minmax(0.05, 1, access)
                element['real_quantity'] = element['capacity'] * Rock.MAX_REAL_QUANTITY_FACTOR
        else:
            self.elements.update(elements)

    def export_encode(self):
        # Update position anchor because on init, Rock sets anchor time to current universe time (hence on import we must have the position at current time)
        self.refresh_anchor()
        e = {**super().export_encode(), **self.__dict__}
        return e

    @classmethod
    def import_decode(cls, uni, d):
        imp = cls(
            # Rock args
            mass=d['mass'],
            fractal_index=d['fractal_index'],
            seed=Seed(*d['seed']),
            elements=d['elements'],
            elements_anchor=d['_elements_anchor'],
            # Body args
            uni=uni, buid=d['buid'],
            subclass=d['_subclass'],
            name=d['name'],
            exists=d['_exists'],
            starting_position=d['_Body__anchor_position'],
            starting_velocity=d['_Body__velocity'],
            vv_context=d['_Body__vv_context'],
            )
        return imp

    def has_element(self, ename):
        return self.elements[ename]['capacity'] > 0

    def refresh_real_quantities(self):
        time_passed = self.uni.time - self._elements_anchor
        if time_passed == 0:
            return
        for ename, e in self.elements.items():
            if not self.has_element(ename):
                continue
            maxq = e['capacity'] * Rock.MAX_REAL_QUANTITY_FACTOR
            if e['real_quantity'] >= maxq:
                continue
            regen = time_passed * e['capacity']
            post_regen = e['real_quantity'] + regen
            e['real_quantity'] = min(post_regen, maxq)
        self._elements_anchor = self.uni.time

    def mine_check(self, ename):
        return self.elements[ename]['capacity'] > 0 and self.elements[ename]['real_quantity'] > 0

    def mine(self, ename, quantity):
        self.refresh_real_quantities()
        if not self.mine_check(ename):
            self.uni.debug(f'{self.fname} requested to mine from saturated element {ename}.')
            return
        final_quantity = quantity * self.elements[ename]['accessibility']
        self.elements[ename]['real_quantity'] -= final_quantity
        return final_quantity

    def get_cap(self, ename):
        return self.elements[ename]['capacity']

    def get_accessibility(self, ename):
        return self.elements[ename]['accessibility']

    def get_realq(self, ename):
        self.refresh_real_quantities()
        return self.elements[ename]['real_quantity']

    @property
    def is_rock(self):
        return True

    @property
    def wet_mass(self):
        return self.mass

    @property
    def dry_mass(self):
        return self.mass

    @property
    def drive_mass(self):
        return self.mass

    @property
    def emissions(self):
        return self.mass

    @property
    def parent(self):
        if self._ancestry[:-1] == []:
            return None
        return self.uni.id2body(self.uni.ancestry[str(self._ancestry[:-1])])


class Ship(Body):
    BASE_SERVICE_TIME = 100000

    def __init__(self, allegiance,
                 docked=None, tugging=None, cargo_hold=None,
                 maintenance_anchor_condition=None,
                 maintenance_anchor_time=None,
                 disintegrate_stack=None,
                 sensor_callback_stack=None, sensor_databank=None,
                 order_context=None,
                 control_params=None,
                 service_stats=None,
                 bridge=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.__allegiance = allegiance
        self._docked = docked
        self._tugging = set() if tugging is None else tugging
        self.player = self.uni.id2player(self.__allegiance)
        # Maintenance
        self.__maintenance_anchor_condition = maintenance_anchor_condition
        self.__maintenance_anchor_time = maintenance_anchor_time
        self.__disintegrate_stack = disintegrate_stack
        if self.__maintenance_anchor_condition is not None:
            if maintenance_anchor_time is None:
                raise ValueError(f'Cannot set maintenance anchor condition without anchor time.')
        else:
            self.set_mcondition(1)
            self.check_disintegrate(context={'summary': 'First check_disintegrate'})
        # Sensor
        self._sensor_callback_stack = Scheduler() if sensor_callback_stack is None else sensor_callback_stack
        self.sensor_databank = collections.defaultdict(lambda: {
            'visible': False,
            'do_check': True,
            'range': None,
            })
        if sensor_databank is not None:
            self.sensor_databank.update(sensor_databank)
        # Cargo hold
        self.fuel_tank = 0
        self.cargo_hold = collections.defaultdict(lambda: 0)
        if cargo_hold is not None:
            self.cargo_hold.update(cargo_hold)
        # Other stats
        self.service_stats = {'fuel_consumed': 0} if service_stats is None else service_stats

        # Bridge
        self.__order_context = {
            'order': '',
            'drive': '',
            'mine': '',
            'build': '',
            } if order_context is None else order_context
        self.control_params = {
            'halt_complete': True,
            'repeat_orders': False,
            'fuel_material': common.E.NAMES[0],
            'fuel_reserve': 0,
            'mine_block_time': 100,
            'halt_enemies': True,
            'dock_block': False,
            }
        if control_params is not None:
            self.control_params.update(control_params)

        # New Bridge
        self.bridge = agency.Bridge(self) if bridge is None else bridge

    def export_encode(self):
        e = {**super().export_encode(), **self.__dict__}
        e['_tugging'] = list(e['_tugging'])
        e['player'] = e['player'].pid
        e['_sensor_callback_stack'] = e['_sensor_callback_stack'].export_encode()
        return e

    @classmethod
    def import_decode(cls, uni, d):
        imp = cls(
            # Ship args
            allegiance=d['_Ship__allegiance'],
            docked=d['_docked'],
            tugging=set(d['_tugging']),
            cargo_hold=d['cargo_hold'],
            maintenance_anchor_condition=d['_Ship__maintenance_anchor_condition'],
            maintenance_anchor_time=d['_Ship__maintenance_anchor_time'],
            sensor_callback_stack=Scheduler.import_decode(d['_sensor_callback_stack']),
            sensor_databank=d['sensor_databank'],
            order_context=d['_Ship__order_context'],
            control_params=d['control_params'],
            service_stats=d['service_stats'],
            # Body args
            uni=uni,
            buid=d['buid'],
            subclass=d['_subclass'],
            name=d['name'],
            exists=d['_exists'],
            starting_position=d['_Body__anchor_position'],
            starting_velocity=d['_Body__velocity'],
            vv_context=d['_Body__vv_context'],
            )
        return imp

    # ORDERS
    def timed_callback(func):
        """
        Wraps functions that may be passed to simsched and will be called arbitrarily.
        This wrapper can prevent irrelevant callbacks from causing chaos.
        """
        @functools.wraps(func)
        def wrapper_timed_callback(self, context, *args, **kwargs):
            if not self.exists:
                return None
            return func(self, context, *args, **kwargs)
        return wrapper_timed_callback

    def add_timed_callback(self, time, callback, context):
        return self.uni.add_timed_callback(time, 'body', self.buid, callback, context)

    def request_guidance(self, message=None):
        m = '.' if message is None else f': {message}'
        self.player.alert(f'{self.fname} requested guidance{m}')
        self.uni.request_simulation_halt()

    def check_partner(exists=True, allied=False):
        """Wraps functions that expect the kwargument named 'target_buid' to be a ship buid.
        This wrapper can prevent trying to interact with uncooperative ships (e.g. dead)."""
        def decorator_check_partner(func, *args, **kwargs):
            @functools.wraps(func)
            def wrapper_check_partner(self, *args, **kwargs):
                assert 'target_buid' in kwargs
                b = self.uni.id2body(kwargs['target_buid'])
                if exists and not b.exists:
                    self.uni.debug(f'{self.fname} checked partner for {func} but partner {b.fname} doesn\'t exist.')
                    return None
                if b.is_ship and (allied and not self.is_ally(b.buid)):
                    self.uni.debug(f'{self.fname} checked partner for {func} but partner {b.fname} isn\'t allied.')
                    return None
                return func(self, *args, **kwargs)
            return wrapper_check_partner
        return decorator_check_partner

    # MAINTENANCE
    @property
    def engineering(self):
        # Ships get free engineering capacity equal to %1 (of dry mass)
        return self.dry_mass/100 + self.stats['engineering']

    @property
    def service_time(self):
        # Service time is proportional to the ratio of enginerring capacity over dry mass
        engineering_ratio = self.engineering / self.dry_mass
        service_time = engineering_ratio * Ship.BASE_SERVICE_TIME
        return service_time

    @property
    def maintenance_condition(self):
        condition = self.__maintenance_anchor_condition - (self.time - self.__maintenance_anchor_time) / self.service_time
        return condition

    @property
    def remaining_service_time(self):
        return self.maintenance_condition * self.service_time

    def set_mcondition(self, value):
        self.__maintenance_anchor_condition = value
        self.__maintenance_anchor_time = self.time
        disintegrate = self.remaining_service_time + PLANK
        context = {
            'summary': f'Expected disintegration for {self.fname} at: {self.time+disintegrate}',
            }
        self.__disintegrate_stack = [self.time+disintegrate, 'check_disintegrate', context]

    def repair_cost(self, percent=1, cap_max=True):
        """To find the repair costs we must convert the condition percent to the equivalent in time. This tells the proportion of our maintenance costs required to repair a certain percent of condition. This is because the maint_costs stat represents cost per annum (1,000,000 units of time)."""
        assert 0 < percent <= 1
        if cap_max:
            percent = min(percent, 1-self.maintenance_condition)
        if percent == 0:
            return {k: 0 for k in self.stats['maint_cost']}
        maint_time = percent * self.service_time
        cost_proportion = maint_time / 1000000
        rcost = {k: v*cost_proportion for k, v in self.stats['maint_cost'].items()}
        return rcost

    @timed_callback
    def check_disintegrate(self, context):
        # Check if we are fine
        if self.maintenance_condition > 0:
            # Update the universe of current disintegration projection
            assert self.__disintegrate_stack[0] > self.uni.time
            self.add_timed_callback(*self.__disintegrate_stack)
            return
        # Otheriwse we disintegrate
        self.player.alert(f'{self.fname} disintegrated from disrepair!')
        self.disintegrate()
        self.uni.request_simulation_halt()

    def disintegrate(self, context=None):
        self.player.alert(f'{self.fname} disintegrated.')
        self.do_kill()

    def do_kill(self):
        self.player.remove_ship(self.buid)
        self.terminate_existence()
        self.name = f'Remains of {self.name}'

    # DOCK/TUG
    def command_dock(self, target_buid):
        t = self.uni.id2body(target_buid)
        if not self.check_dock(target_buid):
            self.uni.debug(f'{self.fname} failed to initiate docking order.')
            return -1
        self.do_dock(target_buid)
        # This took us no time - return None
        return None

    def command_undock(self):
        self.do_undock()
        # This took us no time - return None
        return None

    def check_dock(self, target_buid):
        t = self.uni.id2body(target_buid)
        if target_buid == self.buid:
            self.uni.debug(f'{self.fname} cannot dock to itself.')
            return False
        if self.control_params['dock_block']:
            self.uni.debug(f'{self.fname} bridge control prevented from docking to {t.fname}.')
            return False
        if self.is_docked or t.is_docked:
            self.uni.debug(f'{self.fname} cannot dock to {t.fname} while either is already docked.')
            return False
        if len(self._tugging) > 0:
            self.uni.debug(f'{self.fname} cannot dock to {t.fname} while tugging other ships (tugging: {sjoin(self.uni.id2body(_).fname for _ in self._tugging)})')
            return False
        if not self.matches_vv(target_buid):
            self.uni.debug(f'{self.fname} cannot dock to {t.fname} while velocities don\'t match.')
            return False
        return True

    def do_dock(self, target_buid):
        assert self.check_dock(target_buid)
        t = self.uni.id2body(target_buid)
        self._docked = target_buid
        t._tugging.add(self.buid)
        self.uni.debug(f'{self.fname} docked to {t.fname}')

    def do_undock(self, target_buid=None):
        if target_buid is not None:
            if self._docked != target_buid:
                return
        t = self.uni.id2body(self._docked)
        assert self.buid in t._tugging
        t._tugging.remove(self.buid)
        self._docked = None

    @property
    def tugging_mass(self):
        return sum(self.uni.id2body(tb).wet_mass for tb in self._tugging)

    @property
    def is_docked(self):
        return self._docked is not None

    # REPAIR
    # @check_partner(allied=True)
    def command_serve(self, target_buid, percent):
        if not self.check_repair(target_buid, percent):
            self.uni.debug(f'{self.fname} failed to initiate repair order.')
            return -1
        self.do_repair(target_buid, percent)
        # This took us no time - return None
        return None

    def check_repair(self, target, percent):
        t = self.uni.id2body(target)
        if not self.matches_vv(target):
            self.uni.debug(f'{self.fname} and {t.fname} are too far to repair eachother.')
            return False
        costs = t.repair_cost(percent)
        if DictOp.sum(costs) == 0:
            self.uni.debug(f'{self.fname} cannot repair {t.fname} (already at full repair)')
            return False
        if not self.check_cargo_remove(costs):
            self.uni.debug(f'{self.fname} cannot repair {t.fname} by % {round(percent*100, 1)} for lack of required material: {costs}')
            return False
        return True

    def do_repair(self, target, percent):
        t = self.uni.id2body(target)
        percent = min(percent, 1-t.maintenance_condition)
        costs = t.repair_cost(percent)
        final_cond = t.maintenance_condition + percent
        self.do_cargo_remove(costs)
        t.set_mcondition(final_cond)
        # self.uni.debug(f'{self.fname} repaired {t.fname} for % {round(percent*100, 1)} maintenance to % {round(final_cond*100, 1)} condition.')
        # self.uni.debug(f'Repair cost:{adis(costs)}')

    def order_control(self, control, control_value, v1, compare, v2):
        # if not compare(v1, v2):
        #     return None
        if control == 'wait':
            finish = self.time + control_value
            self.uni.debug(f'{self.name} waiting for {control_value} (until: {finish})')
            return finish
        raise ValueError(f'Unknown control type: {control}')

    # DRIVE
    def order_drive(self, target, speed, breaks=True):
        # Translate target parameter
        if isinstance(target, np.ndarray):
            target_str = f'{dis_coords(target)}'
        if isinstance(target, int):
            target = self.uni.id2body(target).position
            target_str = f'{self.uni.id2body(target).name} ({dis_coords(target)})'
        elif not isinstance(target, np.ndarray):
            target = np.array(target)
        assert isinstance(target, np.ndarray)
        speed = float(speed)
        if not self.drive_check(target, speed, breaks):
            self.uni.debug(f'{self.fname} failed to initiate drive order.')
            return -1

        # Do drive order and retrieve when order is complete
        arrival_time = self.do_drive(target, speed, breaks)
        return arrival_time

    def drive_check(self, target, speed, breaks):
        if self.stats['thrust'] == 0:
            self.uni.debug(f'{self.fname} cannot drive without thrust.')
            return False
        # Check if we are already on the target
        if self.at_target(target):
            self.uni.debug(f'{self.fname} already at target.')
            return False
        # Check if we are docked
        if self.is_docked:
            self.uni.debug(f'{self.fname} cannot drive while docked.')
            return False
        # Speed must be positive
        if speed <= 0:
            self.uni.debug(f'{self.fname} drive speed must be greater than 0.')
            return False
        # Ensure ship has fuel to perform burn(s)
        total_fuel = self.calculate_fuel_costs(target, speed, breaks)
        if self.fuel_tank < total_fuel:
            self.uni.debug(f'{self.fname} not enough fuel for burn (naive check): {self.fuel_tank} / {self.deltav_fuel_cost(total_fuel)} {self.fuel_mat}.')
            return False
        return True

    @property
    def fuel_mat(self):
        return self.control_params['fuel_material']

    @property
    def drive_mass(self):
        return self.wet_mass + self.tugging_mass

    def drive_warmup_time(self, dvmag):
        if self.stats['thrust'] == 0:
            return None
        warmup_time = dvmag * self.drive_mass / self.stats['thrust']
        return warmup_time

    def do_drive(self, target, speed, breaks):
        """Do drive: warmup engine and queue departure burn (with context). Return time until done (warmup + travel)."""
        # Warm up the drives
        target_vector = target - self.position
        new_vv = normalize(target_vector, mag=speed)
        warmup_time = self.drive_warmup_time(vmag(self.vv - new_vv))
        depart_time = self.time + warmup_time
        distance = vmag(target_vector)
        travel_time = distance / speed
        arrival_time = self.time + warmup_time + (travel_time if breaks else 0)
        fuel_cost = self.calculate_fuel_costs(target, speed, breaks)
        context = {
            'context': Seed().r,
            'new_vv': new_vv,
            'target': target,
            'speed': speed,
            'breaks': breaks,
            'fuel_cost': fuel_cost,
            'warmup_time': warmup_time,
            'depart_time': depart_time,
            'distance': distance,
            'travel_time': travel_time,
            'arrival_time': arrival_time,
            'summary': f'{self.fname} driving to {dis_coords(target)}, travel time: {adis(travel_time)}, breaking: {breaks}.',
            }
        # Queue departure_burn when drive has warmed up
        self.__order_context['drive'] = context['context']
        self.uni.debug(f'{self.fname} warmimg up engines @{speed}. Departure in: {warmup_time}, travel_time: {travel_time}.')
        self.uni.debug(adis(context))
        self.add_timed_callback(depart_time, 'do_drive_departure_burn', context)
        return arrival_time

    @timed_callback
    def do_drive_departure_burn(self, context):
        # Discard past context
        if self.__order_context['drive'] != context['context']:
            self.uni.debug(f'{self.fname} discarded departure_burn (past context).\n{adis(context)}')
            return
        # Abort if drive check fails
        if not self.drive_check(context['target'], context['speed'], context['breaks']):
            self.debug(f'{self.fname} failed to perform departure burn.')
            return
        # Departure burn
        self._drive_burn(context['new_vv'])
        self.uni.debug(f'{self.fname} performed departure burn.')
        # If breaking is requested, we must queue up breaking burn
        if not context['breaks']:
            # Clear drive_context as we are done
            self.__order_context['drive'] = ''
            return
        self.add_timed_callback(context['arrival_time'], 'do_drive_breaking_burn', context)

    @timed_callback
    def do_drive_breaking_burn(self, context):
        # Discard past context
        if self.__order_context['drive'] != context['context']:
            self.uni.debug(f'{self.fname} discarded departure_burn (past context).\n{adis(context)}')
            return
        assert context['breaks']
        # Clear drive_context as we are done
        self.__order_context['drive'] = ''
        # Perform the burn
        self._drive_burn(None)
        self.uni.debug(f'{self.fname} completed drive.', force_print=True)
        # self.uni.debug(f'{adis(context)}')

    def _drive_burn(self, vv):
        """Drive to target point using engine."""
        # vv of None means break (halt velocity)
        if vv is None:
            vv = np.zeros(2)
        # Remove fuel based on the Tsiolkovsky rocket equation
        dv = vmag(self.vv - vv)
        fuel_cost = self.deltav_fuel_cost(dv)
        assert self.fuel_tank >= fuel_cost
        self.fuel_tank -= fuel_cost
        # Set our new velocity vector
        self.set_vv(vv)
        # Set velocity of all tugged ships
        for tugged_buid in self._tugging:
            self.uni.id2body(tugged_buid).set_vv(vv)
        self.service_stats['fuel_consumed'] += fuel_cost

    def calculate_fuel_costs(self, target, speed, breaks):
        vv_target = target-self.position
        if vmag(vv_target) < PLANK:
            return 0
        new_vv = normalize(vv_target, mag=speed)
        departure_burn = vmag(self.vv - new_vv)
        breaking_burn = speed if breaks else 0
        return self.deltav_fuel_cost(departure_burn + breaking_burn)

    def deltav_fuel_cost(self, dv, use_mass=None):
        assert isinstance(dv, float) or isinstance(dv, int)
        if self.stats["thrust"] == 0:
            return None
        mass = self.drive_mass if use_mass is None else use_mass
        return mass * dv / self.stats['isp'] / 1000

    @property
    def available_dv(self):
        if self.stats['thrust'] == 0:
            return 0
        return self.fuel_tank / self.deltav_fuel_cost(1)

    @property
    def total_dv(self):
        if self.stats['thrust'] == 0:
            return 0
        return self.stats['fuel_tank'] / self.deltav_fuel_cost(1, use_mass=self.dry_mass+self.stats['fuel_tank'])

    def matches_vv(self, target_buid):
        return math.fabs(vmag(self.vv - self.uni.id2body(target_buid).vv)
                         ) < self.uni.proximity_distance

    def at_target(self, target):
        return math.fabs(vmag(self.position - target)) < self.uni.proximity_distance

    @property
    def is_stationary(self):
        return vmag(self.vv) < PLANK_LENGTH

    def can_reach(self, target, speed, include_return=False):
        to = self.calculate_fuel_costs(target, speed, True)
        fro = self.deltav_fuel_cost(speed*2)
        return to+fro < self.fuel_tank

    # BUILD
    def order_build(self, target_cuid):
        """Build the target ship class."""
        if not self.build_check(target_cuid):
            self.uni.debug('Cannot build...')
            return -1

        # Set a callback for when building is complete
        subclass = self.uni.subclasses[target_cuid]
        stats = subclass['stats']
        bc = self.__order_context['build'] = self.uni.seed.randfloat()
        # Build capacity is in units per kd, hence we divide by 1000
        build_time = stats['build_cost'] / (self.stats['build_capacity'] / 1000)

        build_complete = self.time + build_time
        self.do_cargo_remove(subclass['stats']['material_cost'])

        context = {
            'build_context': bc,
            'new_ship_name': f'New ship {Seed().randstr(4)}',
            'target_cuid': target_cuid,
            'material_cost': stats['material_cost'],
            'build_time': build_time,
            'build_complete': build_complete,
            'summary': f'Building: {subclass["name"]} (complete @{build_complete})',
            }
        self.add_timed_callback(build_complete, 'order_build_conclude', context)

        # return build complete time
        return build_complete

    def build_check(self, target_cuid):
        if self.stats['build_capacity'] == 0:
            self.uni.debug(f'{self.fname} cannot build without build capacity.')
            return False
        if target_cuid not in self.uni.subclasses or target_cuid not in self.player.ship_designs:
            self.uni.debug(f'{self.fname} cannot find target ship class {target_cuid} to build.')
            return False
        stats = self.uni.subclasses[target_cuid]['stats']
        if not self.check_cargo_remove(stats['material_cost']):
            self.uni.debug(f'{self.fname} missing material for build cost of: {adis(stats["material_cost"])}')
            return False
        return True

    @timed_callback
    def order_build_conclude(self, context):
        # Discard invalid concludes
        if context['build_context'] != self.__order_context['build'] or context['build_complete'] != self.time:
            return
        target_cuid = context['target_cuid']
        subclass = self.uni.subclasses[target_cuid]
        ship_name = context['new_ship_name']
        self.player.new_ship(self.uni.new_ship(name=ship_name,
                                               subclass=target_cuid,
                                               starting_position=self.position,
                                               allegiance=self.__allegiance))
        self.player.alert(f'{self.fname} built new ship {ship_name} of class {subclass["name"]}')

    # CARGO
    @check_partner(allied=True)
    def order_refuel(self, target_buid, count=0):
        t = self.uni.id2body(target_buid)
        if not self.matches_vv(target_buid):
            self.uni.debug(f'Must match velocities to transfer cargo between {self.fname} and {t.fname}')
            return -1
        # Determine the fuel/cargo to transfer
        item = self.fuel_mat
        available_tank = self.stats['fuel_tank'] - self.fuel_tank
        # 0 count represents the maximum amount possible
        count = float('inf') if count <= 0 else count
        assert count > 0
        count = min(count, available_tank, t.cargo_hold[item])
        # Assemble the cargo dictionary
        cargo = {item: count}
        cargo_mass = common.cargo_mass(cargo)
        # Transfer the cargo
        assert t.check_cargo_remove(cargo)
        t.do_cargo_remove(cargo)
        self.fuel_tank += count
        # Return that order requires no time to complete
        return None

    @check_partner(allied=True)
    def order_transfer_cargo(self, target_buid, item, count, take):
        if target_buid == self.buid:
            self.uni.debug(f'{self.fname} aborted orders to transfer cargo with itself.')
            return -1
        t = self.uni.id2body(target_buid)
        if not self.matches_vv(target_buid):
            self.uni.debug(f'Must match velocities to transfer cargo between {self.fname} and {t.fname}')
            return -1
        taker = self if take else t
        giver = t if take else self
        # Determine the cargo to transfer
        # 0 count represents the maximum amount possible to transfer
        count = float('inf') if count <= 0 else count
        assert count > 0
        count = min(count, giver.cargo_hold[item], taker.remaining_hold)
        # Assemble the cargo dictionary
        cargo = {item: count}
        cargo_mass = common.cargo_mass(cargo)
        # Transfer the cargo
        giver.do_give_cargo(taker.buid, cargo)
        # Return that order requires no time to complete
        return None

    def do_give_cargo(self, target_buid, transfer_cargo):
        target = self.uni.id2body(target_buid)
        # Do checks, we have cargo to send and target can take it
        if not self.check_cargo_remove(transfer_cargo):
            self.uni.debug(f'{self.fname} trying to transfer cargo to {target.fname}, but does not have said cargo:\n{disd(transfer_cargo)}')
            return
        if not target.check_cargo_add(transfer_cargo):
            self.uni.debug(f'{self.fname} trying to transfer cargo to {target.fname}, but target cannot take.')
            return
        self.do_cargo_remove(transfer_cargo)
        target.do_cargo_add(transfer_cargo)
        self.uni.debug(f'{self.fname} transfered {common.cargo_mass(transfer_cargo)} mass of cargo to {target.fname}')
        self.uni.debug('Transfer cargo:'+adis(transfer_cargo))
        self.uni.debug(f'{self.fname} cargo:'+adis(self.cargo_hold))
        self.uni.debug(f'{target.fname} cargo:'+adis(target.cargo_hold))

    def do_cargo_remove(self, cargo):
        for item, amount in cargo.items():
            self.cargo_hold[item] -= amount
            assert self.cargo_hold[item] >= 0

    def do_cargo_add(self, cargo):
        for item, amount in cargo.items():
            self.cargo_hold[item] += amount

    def check_cargo_remove(self, cargo):
        for item, amount in cargo.items():
            if self.cargo_hold[item] < amount:
                return False
        return True

    def check_cargo_add(self, transfer_cargo):
        return common.cargo_mass(transfer_cargo) <= self.remaining_hold

    @property
    def cargo_mass(self):
        return common.cargo_mass(self.cargo_hold)

    @property
    def remaining_hold(self):
        return self.stats['hold'] - self.cargo_mass

    # MINE
    def order_mine(self, target_buid, target_element):
        """Mine the target element using miner."""
        if not self.check_order_mine(target_buid, target_element):
            self.uni.debug(f'{self.fname} failed to initiate mining order.')
            return -1
        tb = self.uni.id2body(target_buid)

        # Set a callback for when mining is complete
        mc = self.__order_context['mine'] = self.uni.seed.randfloat()
        block_time = self.control_params['mine_block_time']
        block_complete = self.time + block_time
        context = {
            'mine_context': mc,
            'target_buid': target_buid,
            'target_element': target_element,
            'block_time': block_time,
            'block_complete': block_complete,
            'block_size': tb.get_accessibility(target_element) * block_time,
            'summary': f'{self.fname} Mining: {target_element}',
            }
        self.add_timed_callback(block_complete, 'order_mine_conclude', context)

        # return mine block complete time
        return block_complete

    def check_order_mine(self, target_buid, target_element):
        if self.stats['mining_capacity'] == 0:
            self.uni.debug(f'{self.fname} cannot mine without mining capacity.')
            return False
        if not self.matches_vv(target_buid):
            self.uni.debug(f'{self.fname} must match velocity with {target_buid} to mine.')
            return False
        target_body = self.uni.id2body(target_buid)
        if not target_body.is_rock:
            self.uni.debug(f'{self.fname} mine target {target_body.fname} is not a rock.')
            return False
        distance = self.uni.get_body_distance(self.buid, target_buid)
        if distance > PROXIMITY_DISTANCE:
            self.uni.debug(f'{self.fname} mine target {target_body.fname} too far to mine ({distance}).')
            return False
        # if target_body.get_cap(target_element) <= 0:
            self.uni.debug(f'{self.fname} cannot find specified element {target_element} in target {target_body.fname}')
        #     return False
        if not target_body.mine_check(target_element):
            # self.uni.debug(f'{self.fname} cannot mine - missing or oversaturated element {target_element} from {target_body.fname} (cap:{target_body.get_cap(target_element)} realq:{target_body.get_realq(target_element)})')
            return False
        return True

    @timed_callback
    def order_mine_conclude(self, context):
        # Discard invalid concludes
        if context['mine_context'] != self.__order_context['mine']:
            return
        self.__order_context['mine'] = ''
        target_buid, target_element = context['target_buid'], context['target_element']
        if not self.check_order_mine(context['target_buid'], context['target_element']):
            self.uni.debug(f'{self.fname} failed to complete mining.\n{adis(context)}')
            return
        tb = self.uni.id2body(target_buid)
        # Mining capacity is in units per kd, hence we divide by 1000
        effort = context['block_time'] * self.stats['mining_capacity'] / 1000
        block_size = tb.get_accessibility(target_element) * effort
        block = {target_element: block_size}
        if not self.check_cargo_add(block):
            self.request_guidance(f'no room for mined material {block} - discarding block.')
            return
        actual_block_size = tb.mine(target_element, effort)
        assert actual_block_size - block_size < PLANK
        self.do_cargo_add(block)
        # self.uni.debug(f'{self.fname} mined {target_element} x{block_size} from {tb.fname}')

    # SENSOR
    @property
    def has_sensor(self):
        return self.stats['sensitivity'] > 0

    def sensor_range(self, target):
        """Cached value of the range of our sensor to target"""
        if not self.sensor_databank[target]['do_check']:
            return None
        return self.sensor_databank[target]['range']

    def calculate_sensor_range(self, target_buid):
        """Calculate the particular range of our sensors to target"""
        assert self.has_sensor
        return common.Sensing.sensor_range(
            mass=self.uni.id2body(target_buid).dry_mass,
            sens=self.stats['sensitivity'],
            )

    @timed_callback
    def sensor_change(self, context):
        """
        Handles sensor events queued by self.sensor_stack_add().
        These events represent a body moving in or out of our sensor envelope.
        """
        # Assert that this callback was known to us and it was the next event in our sensor stack. Remove it.
        assert self._sensor_callback_stack.next_event[2]['rid'] == context['rid']
        self._sensor_callback_stack.pop_next()
        # If the next event in the sensor stack is unknown to the universe, give it - this is critical for the sensor stack to work.
        if len(self._sensor_callback_stack) > 0:
            # Check if the universe already knows about our next sensor stack event
            if not self._sensor_callback_stack.next_event[-1]:
                # Our sensor stack event also retains an extra item representing whether the universe is aware or not, so we omit that before passing to universe
                self.add_timed_callback(*self._sensor_callback_stack.next_event[:-1])
                # Remember that the universe now knows about the next event
                self._sensor_callback_stack.entries[0] = (*self._sensor_callback_stack.entries[0][:-1], True)
        # If our velocity has changed since this event, it is obsolete.
        if context['vv_context'] != self.vvcon:
            # self.uni.debug(f'Discarding obsolete sensor change: {adis(context)}')
            return
        target_buid = context['target_buid']
        target_body = self.uni.id2body(target_buid)
        range = self.sensor_range(target_buid)
        # Only check objects we are tracking (does not include discovered inanimate objects).
        if range is not None:
            # One more check to confirm that we are at the expected distance.
            # (Events may become still obsolete without our own vv changing)
            # And then finally call the method for handling this event.
            if (range - PLANK_LENGTH) < self.uni.get_body_distance(self.buid, target_buid) < (range + PLANK_LENGTH):
                if context['entering']:
                    self.sensor_in(target_buid)
                else:
                    self.sensor_out(target_buid)

    def sensor_in(self, target_buid):
        """Target is entering our sensor range"""
        tb = self.uni.id2body(target_buid)

        # If body is inanimate, we now know its location indefinitely and don't need to check distances
        if tb.is_rock:
            self.sensor_databank[target_buid]['do_check'] = False
            # Alert the player if needed
            if target_buid not in self.player.rock_database:
                self.player.new_rock(target_buid)
                self.player.alert(f'{self.fname} discovered a new rock: {tb.fname}')
        # Enemey ships are handled separately
        elif tb.is_ship and self.is_player and not tb.is_player:
            # Halt for enemies if needed
            if self.control_params['halt_enemies']:
                self.uni.request_simulation_halt()
            # Alert the player if needed
            if target_buid not in self.player.visible_buids:
                self.player.alert(f'{self.fname} found a new ship: {tb.fname}')
        # Finally, change the targets visibility to true.
        self.sensor_databank[target_buid]['visible'] = True

    def sensor_out(self, target_buid):
        """Target is leaving our sensor range"""
        target_body = self.uni.id2body(target_buid)
        # Remove enemy ships from visibility
        if target_body.is_ship:
            self.sensor_databank[target_buid]['visible'] = False

    def is_visible(self, target_buid):
        return self.sensor_databank[target_buid]['visible']

    @property
    def visible_bodies(self):
        return [*filter(lambda x: self.sensor_databank[x]['visible'], self.sensor_databank.keys())]

    @property
    def visible_ships(self):
        return [*filter(lambda x: self.uni.id2body(x).is_ship, self.visible_bodies)]

    # Survey
    def order_survey(self, target_buid):
        """Survey the target rock."""
        if not self.check_order_survey(target_buid):
            return -1
        tb = self.uni.id2body(target_buid)

        # Set a callback for when survey is complete
        sc = self.__order_context['survey'] = self.uni.seed.randfloat()
        survey_time = tb.resolve_subclass['survey_cost'] / (self.stats['survey'] / 1000)  # survey stat is per kd
        survey_complete = self.time + survey_time
        context = {
            'survey_context': sc,
            'target_buid': target_buid,
            'survey_time': survey_time,
            'survey_complete': survey_complete,
            'summary': f'{self.fname} surveying: {tb.fname}',
            }
        self.add_timed_callback(survey_complete, 'order_survey_conclude', context)

        # return mine block complete time
        return survey_complete

    def check_order_survey(self, target_buid):
        if self.stats['survey'] == 0:
            self.uni.debug(f'{self.fname} cannot survey without survey components.')
            return False
        if not self.matches_vv(target_buid):
            self.uni.debug(f'{self.fname} must match velocity with {target_buid} to survey.')
            return False
        target_body = self.uni.id2body(target_buid)
        if not target_body.is_rock:
            self.uni.debug(f'{self.fname} survey target {target_body.fname} is not a rock.')
            return False
        if self.player.rock_database[target_buid]['survey']:
            self.uni.debug(f'{self.fname} survey target {target_body.fname} is already surveyed.')
            return False
        return True

    @timed_callback
    def order_survey_conclude(self, context):
        # Discard invalid concludes
        if context['survey_context'] != self.__order_context['survey']:
            return
        self.__order_context['survey'] = ''
        target_buid = context['target_buid']
        if not self.check_order_survey(target_buid):
            self.uni.debug(f'{self.fname} failed to complete survey.\n{adis(context)}')
            return
        tb = self.uni.id2body(target_buid)
        self.player.rock_database[target_buid]['survey'] = True
        self.player.alert(f'{self.name} surveyed {tb.name}.')


    # Universe interface
    def new_body(self, target_buid):
        """This allows a ship to prepare anything as a new body enters its sector"""
        if target_buid == self.buid:
            return
        # Add body to sensor range database
        # TODO prune inanimate objects already discovered by other ships (permanently visible to player)
        if self.has_sensor:
            sensor_range = self.calculate_sensor_range(target_buid)
            self.sensor_databank[target_buid]['range'] = sensor_range
            target = self.uni.id2body(target_buid)
            self.uni.debug(f'{self.fname} new range for {target.fname}: {adis(sensor_range)} (mass: {adis(target.dry_mass)})')
            if self.uni.get_body_distance(self.buid, target_buid) - PLANK_LENGTH <= sensor_range:
                self.sensor_in(target_buid)
            self.check_velocity_interactions(target_buid)

    def old_body(self, target_buid):
        """This allows a ship to conclude anything as an old body leaves its sector"""
        # Remove body from sensor range database
        if target_buid in self.sensor_databank:
            del self.sensor_databank[target_buid]

    def set_vv(self, *args, **kwargs):
        """Set the velocity vector of this ship. We clear our sensor stack as it is full of irrelevant velocity interaction events (such as sensor in and out)."""
        self.sensor_clear_stack()
        super().set_vv(*args, **kwargs)

    def check_velocity_interactions(self, target_buid):
        """Called when our or target's velocity vector changes and allows for this ship to react to other ships movements (by queueing events to simsched for the appropriate time).
        In particular, this applies to sensors which will change 'visibility' of other ships when they enter and leave our sensory envelope (based on their distance). These generated events are handled by our sensor stack - see self.sensor_stack_add()."""
        if self.has_sensor and not self.is_ally(target_buid):
            sensor_range = self.sensor_range(target_buid)
            if sensor_range is not None:
                # Find when the range of time in which the target is in our sensor range
                sensor_time_range = self.uni.when_at_range(delta=sensor_range, b1=self.buid, b2=target_buid)
                if sensor_time_range is not None:
                    assert len(sensor_time_range) == 2
                    # We found the time at which target enters and exits our sensor range
                    # (including in the past, as newtonian physics work equally forward and backward in time)
                    for _state, time in enumerate(sensor_time_range):
                        # We prune time that are in the past
                        if time >= 0:
                            time += self.uni.time
                            context={
                                'vv_context': self.vvcon,
                                'rid': Seed().r,
                                'target_buid': target_buid,
                                'entering': not _state,
                                'summary': f'{self.fname} {"no longer sees" if _state else "now sees"} {self.uni.id2body(target_buid).fname} (@ {sensor_time_range[0] + self.uni.time} to {sensor_time_range[1] + self.uni.time})',
                                }
                            # Finally, we hand over the event to sensor_stack_add to handle scheduling in simsched.
                            self.sensor_stack_add((time, 'sensor_change', context))

    def sensor_clear_stack(self):
        """
        We only clear the events the universe doesn't already know about. Since the universe will call them anyway, we can use this knowledge to prevent queueing events that don't need to be added yet.
        """
        events = []
        for e in self._sensor_callback_stack.entries:
            # Remember this entry if the universe knows about it
            if e[-1]:
                events.append(e)
        self._sensor_callback_stack.clear()
        for e in events:
            self._sensor_callback_stack.add_event_sorted(e)

    def sensor_stack_add(self, new_e):
        """
        Add new sensor_change event to our sensor stack. This is a stack of events of bodies moving in and out of our sensor envelope. The universe simsched should always know about the first event in our sensor stack. Every time the universe calls for the next event in our sensor stack, we can hand it the next one - see self.sensor_change() (the method called by these events).
        Assuming this, we can safely prune the rest of the events if necessary since the universe will always know about the next event.
        If this new event happens before the closest callback in our stack, then it should be known to the universe. If this new event happens after the closest callback in our stack, then it should be unknown to the universe.
        """
        # If new_e happens before our first item in the stack, let the universe know about it
        uni_needs_know = True
        if len(self._sensor_callback_stack) > 0:
            if new_e[0] > self._sensor_callback_stack.next_event[0]:
                uni_needs_know = False
        # We keep track of this event in our stack
        self._sensor_callback_stack.add_event_sorted((*new_e, uni_needs_know))
        # Let the universe know if needed
        if uni_needs_know:
            self.add_timed_callback(*new_e)

    # Basic stats
    @property
    def stats(self):
        return self.uni.subclasses[self._subclass]['stats']

    @property
    def is_ship(self):
        return True

    @property
    def is_player(self):
        return self.__allegiance is 0

    @property
    def dry_mass(self):
        return self.stats['mass']

    @property
    def mass(self):
        return self.dry_mass

    @property
    def wet_mass(self):
        return self.dry_mass + self.cargo_mass + self.fuel_tank

    @property
    def full_wet_mass(self):
        return self.dry_mass + self.stats['hold'] + self.stats['fuel_tank']

    @property
    def emissions(self):
        return self.dry_mass

    @property
    def allegiance(self):
        return self.__allegiance

    def is_ally(self, target):
        tb = self.uni.id2body(target)
        return tb.is_ship and self.allegiance == tb.allegiance

    @property
    def tags(self):
        return self.bridge.tags


class Scheduler:
    def __init__(self, entries=None, is_sorted=None):
        self.entries = [] if entries is None else entries
        self._is_sorted = True if entries is None else False if is_sorted is None else is_sorted

    def export_encode(self):
        e = {**self.__dict__}
        return e

    @classmethod
    def import_decode(cls, d):
        imp = cls(entries=d['entries'], is_sorted=d['_is_sorted'])
        return imp

    def clear(self):
        self.entries = []
        self._is_sorted = True

    def add_event(self, event):
        self.append(event)
        self._is_sorted = False

    def add_event_sorted(self, new_event):
        if len(self.entries) == 0:
            self.entries.append(new_event)
            return

        self.entries.insert(0, new_event)
        self.sort_events()
        # TODO use binary search to insert instead of sorting

    def sort_events(self):
        self.entries = sorted(self.entries, key=self._sort_key)
        self._is_sorted = True

    def pop_next(self):
        if not self._is_sorted:
            self.sort_events()
        return self.entries.pop(0)

    @property
    def next_event(self):
        return self.entries[0]

    def _sort_key(self, x):
        return x[0]

    def __len__(self):
        return len(self.entries)


class DUEncoder(json.JSONEncoder):
    def default(self, obj):
        def debug(o):
            print(sjoin([
                make_title(f'{o.__class__.__name__}'),
                f'{repr(o)}: {o}',
                ]))
        # debug(obj)
        if isinstance(obj, np.ndarray):
            return tuple((obj[_] for _ in range(len(obj))))
        elif isinstance(obj, set):
            return [_ for _ in obj]
        elif isinstance(obj, RateCounter):
            return None
        log_disk(LOG_PATH, f'JSON Encoding:{vdebug(obj, print_=False)}')
        if isinstance(obj, Rock):
            return obj.export_encode()
        elif isinstance(obj, Ship):
            return obj.export_encode()
        elif isinstance(obj, Scheduler):
            return obj.export_encode()
        elif isinstance(obj, agency.OrderBook):
            return obj.export_encode()
        elif isinstance(obj, agency.Player):
            return obj.export_encode()
        elif isinstance(obj, Seed):
            return (obj.seed, obj.resolution, obj.last_val)
        elif isinstance(obj, Universe):
            # Many other objects refer to the universe object and we would like to avoid circular reference. We always delete these entries in their respective export_encode() method, however the json encoder fails if we dont return anything serializable.
            return None
        return json.JSONEncoder.default(self, obj)


def vmag(a):
    return np.linalg.norm(a)


def normalize(a, mag=1):
    return a / np.linalg.norm(a) * mag


def dis_coords(coords):
    return f'{disn(coords[0], force_scientific=True)},{disn(coords[1], force_scientific=True)}'
