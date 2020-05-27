from link_bot_pycommon.link_bot_scenario import LinkBotScenario


def get_scenario(scenario_name: str):
    if scenario_name == 'link_bot':
        return LinkBotScenario()
    else:
        raise NotImplementedError()
