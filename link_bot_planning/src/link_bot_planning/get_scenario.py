from link_bot_planning.link_bot_scenario import LinkBotScenario
from link_bot_planning.tether_scenario import TetherScenario


def get_scenario(scenario_name: str):
    if scenario_name == 'link_bot':
        return LinkBotScenario()
    elif scenario_name == 'tether':
        return TetherScenario()
    else:
        raise NotImplementedError()
