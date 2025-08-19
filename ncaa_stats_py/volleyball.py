# Author: Joseph Armstrong (armstrongjoseph08@gmail.com)
# File Name: `volleyball.py`
# Purpose: Houses functions that allows one to access NCAA volleyball data
# Creation Date: 2024-09-20 08:15 PM EDT
# Update History:
# - 2024-09-20 08:15 PM EDT
# - 2025-01-04 03:00 PM EDT
# - 2025-01-18 02:44 PM EDT
# - 2025-02-01 02:40 PM EDT
# - 2025-02-05 08:50 PM EDT
# - 2025-06-12 10:00 AM EDT


import logging
import re
from datetime import date, datetime
from os import mkdir
from os.path import exists, expanduser, getmtime

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser
from pytz import timezone
from tqdm import tqdm

from ncaa_stats_py.helpers.volleyball import _volleyball_pbp_helper
from ncaa_stats_py.utls import (
    _format_folder_str,
    _get_schools,
    _get_webpage,
    _name_smother,
)


class VolleyballConfig:
    """Configuration settings for volleyball data loading."""
    
    def __init__(self):
        self.default_sport = "women"  # Default to women's volleyball
        self.cache_duration_days = 1   # How long to cache data
        self.enable_progress_bars = True
        self.auto_retry_failed_requests = True
        
    def set_default_sport(self, sport: str):
        """Set default sport for all volleyball functions."""
        valid_sports = ["women", "men", "w", "m", "wvb", "mvb"]
        if sport.lower() not in valid_sports:
            raise ValueError(f"Sport must be one of: {valid_sports}")
        self.default_sport = sport.lower()


# Global config instance
volleyball_config = VolleyballConfig()


def configure_volleyball(default_sport: str = "women", **kwargs):
    """
    Configure default settings for volleyball functions.
    
    Parameters
    ----------
    default_sport : str, default "women"
        Default sport for volleyball functions: "women" or "men"
    **kwargs
        Additional configuration options
        
    Examples
    --------
    >>> configure_volleyball("women")  # Default to women's volleyball
    >>> configure_volleyball("men")    # Default to men's volleyball
    """
    volleyball_config.set_default_sport(default_sport)
    for key, value in kwargs.items():
        if hasattr(volleyball_config, key):
            setattr(volleyball_config, key, value)


def _validate_volleyball_inputs(season: int, level: str | int, sport: str):
    """Validate common volleyball function inputs."""
    # Validate season
    current_year = datetime.now().year
    if season < 2010 or season > current_year + 2:
        raise ValueError(f"Season must be between 2010 and {current_year + 2}")
    
    # Validate level
    valid_levels = [1, 2, 3, "I", "II", "III", "i", "ii", "iii", "D1", "D2", "D3"]
    if level not in valid_levels:
        raise ValueError(f"Level must be one of: {valid_levels}")
    
    # Validate sport
    valid_sports = ["women", "men", "w", "m", "wvb", "mvb"]
    if sport.lower() not in valid_sports:
        raise ValueError(
            f"Sport must be one of: {valid_sports}. "
            f"Use 'women' (default) for women's volleyball or 'men' for men's volleyball."
        )


def _get_sport_params(sport: str):
    """Convert sport string to internal parameters."""
    if sport.lower() in ["women", "w", "wvb"]:
        return False, "WVB"  # get_mens_data=False, sport_id="WVB"
    elif sport.lower() in ["men", "m", "mvb"]:
        return True, "MVB"   # get_mens_data=True, sport_id="MVB"
    else:
        raise ValueError(f"Invalid sport: {sport}")


def get_volleyball_teams(
    season: int,
    level: str | int,
    sport: str = None
) -> pd.DataFrame:
    """
    Retrieves a list of volleyball teams from the NCAA.

    Parameters
    ----------
    season : int
        Required argument.
        Specifies the season you want NCAA volleyball team information from.

    level : int | str
        Required argument.
        Specifies the level/division you want NCAA volleyball team information from.
        This can either be an integer (1-3) or a string ("I"-"III").

    sport : str, optional
        Sport type: "women"/"w"/"wvb" for women's volleyball (default),
        "men"/"m"/"mvb" for men's volleyball.
        If None, uses the configured default.

    Examples
    --------
    # Get all D1 women's volleyball teams for the 2024 season (most common)
    >>> df = get_volleyball_teams(2024, 1)
    
    # Get all D1 men's volleyball teams for the 2024 season
    >>> df = get_volleyball_teams(2024, 1, sport="men")
    
    # Get all D2 women's volleyball teams for the 2023 season
    >>> df = get_volleyball_teams(2023, "II")
    
    # Get all D3 women's volleyball teams for the 2022 season
    >>> df = get_volleyball_teams(2022, 3)

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame object with a list of college volleyball teams
        in that season and NCAA level.
    """
    if sport is None:
        sport = volleyball_config.default_sport
    
    _validate_volleyball_inputs(season, level, sport)
    get_mens_data, sport_id = _get_sport_params(sport)
    
    try:
        return _get_volleyball_teams_internal(season, level, get_mens_data)
    except ConnectionError as e:
        raise ConnectionError(
            f"Failed to connect to NCAA stats website. "
            f"Please check your internet connection and try again. "
            f"Original error: {e}"
        )
    except Exception as e:
        sport_name = "men's" if get_mens_data else "women's"
        raise Exception(
            f"Unexpected error loading {sport_name} volleyball teams for {season} season, level {level}. "
            f"Please report this issue with the following details: {e}"
        )


def get_womens_volleyball_teams(season: int, level: str | int) -> pd.DataFrame:
    """
    Convenience function to get women's volleyball teams.
    
    Parameters
    ----------
    season : int
        The season year (e.g., 2024)
    level : str | int  
        NCAA division: 1/"I", 2/"II", 3/"III"
        
    Examples
    --------
    >>> df = get_womens_volleyball_teams(2024, 1)  # D1 women's volleyball
    >>> df = get_womens_volleyball_teams(2023, "II")  # D2 women's volleyball
    
    Returns
    -------
    pd.DataFrame
        Women's volleyball teams for the specified season and level.
    """
    return get_volleyball_teams(season, level, sport="women")


def get_mens_volleyball_teams(season: int, level: str | int) -> pd.DataFrame:
    """
    Convenience function to get men's volleyball teams.
    
    Parameters
    ----------
    season : int
        The season year (e.g., 2024)
    level : str | int  
        NCAA division: 1/"I", 3/"III" (note: no D2 men's volleyball)
        
    Examples
    --------
    >>> df = get_mens_volleyball_teams(2024, 1)  # D1 men's volleyball
    >>> df = get_mens_volleyball_teams(2023, "III")  # D3 men's volleyball
    
    Returns
    -------
    pd.DataFrame
        Men's volleyball teams for the specified season and level.
    """
    return get_volleyball_teams(season, level, sport="men")


def _get_volleyball_teams_internal(
    season: int,
    level: str | int,
    get_mens_data: bool = False
) -> pd.DataFrame:
    """
    Internal function for retrieving volleyball teams.
    This maintains the original implementation logic.
    """
    sport_id = ""
    load_from_cache = True
    home_dir = expanduser("~")
    home_dir = _format_folder_str(home_dir)
    teams_df = pd.DataFrame()
    teams_df_arr = []
    temp_df = pd.DataFrame()
    formatted_level = ""
    ncaa_level = 0

    if get_mens_data is True:
        sport_id = "MVB"
        stat_sequence = 528
    elif get_mens_data is False:
        sport_id = "WVB"
        stat_sequence = 48

    if isinstance(level, int) and level == 1:
        formatted_level = "I"
        ncaa_level = 1
    elif isinstance(level, int) and level == 2:
        formatted_level = "II"
        ncaa_level = 2
    elif isinstance(level, int) and level == 3:
        formatted_level = "III"
        ncaa_level = 3
    elif isinstance(level, str) and (
        level.lower() == "i" or level.lower() == "d1" or level.lower() == "1"
    ):
        ncaa_level = 1
        formatted_level = level.upper()
    elif isinstance(level, str) and (
        level.lower() == "ii" or level.lower() == "d2" or level.lower() == "2"
    ):
        ncaa_level = 2
        formatted_level = level.upper()
    elif isinstance(level, str) and (
        level.lower() == "iii" or level.lower() == "d3" or level.lower() == "3"
    ):
        ncaa_level = 3
        formatted_level = level.upper()

    if exists(f"{home_dir}/.ncaa_stats_py/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/")

    if exists(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/")

    if exists(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/teams/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/teams/")

    if exists(
        f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/teams/"
        + f"{season}_{formatted_level}_teams.csv"
    ):
        teams_df = pd.read_csv(
            f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/teams/"
            + f"{season}_{formatted_level}_teams.csv"
        )
        file_mod_datetime = datetime.fromtimestamp(
            getmtime(
                f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/teams/"
                + f"{season}_{formatted_level}_teams.csv"
            )
        )
    else:
        file_mod_datetime = datetime.today()
        load_from_cache = False

    now = datetime.today()

    age = now - file_mod_datetime

    if (
        age.days > 1 and
        season >= (now.year - 1) and
        now.month <= 7
    ):
        load_from_cache = False
    elif age.days >= 35:
        load_from_cache = False

    if load_from_cache is True:
        return teams_df

    logging.warning(
        f"Either we could not load {season} D{level} schools from cache, "
        + "or it's time to refresh the cached data."
    )
    schools_df = _get_schools()

    # Volleyball
    if sport_id == "MVB":
        url = (
            "https://stats.ncaa.org/rankings/change_sport_year_div?"
            + f"academic_year={season}.0&division={ncaa_level}.0" +
            f"&sport_code={sport_id}"
        )
    elif sport_id == "WVB":
        url = (
            "https://stats.ncaa.org/rankings/change_sport_year_div?"
            + f"academic_year={season+1}.0&division={ncaa_level}.0" +
            f"&sport_code={sport_id}"
        )

    response = _get_webpage(url=url)

    soup = BeautifulSoup(response.text, features="lxml")
    ranking_periods = soup.find("select", {"name": "rp", "id": "rp"})
    ranking_periods = ranking_periods.find_all("option")

    rp_value = 0
    found_value = False

    while found_value is False:
        for rp in ranking_periods:
            if "final" in rp.text.lower():
                rp_value = rp.get("value")
                found_value = True
                break
            elif "-" in rp.text.lower():
                pass
            else:
                rp_value = rp.get("value")
                found_value = True
                break

    if sport_id == "MVB":
        url = (
            "https://stats.ncaa.org/rankings/institution_trends?"
            + f"academic_year={season}.0&division={ncaa_level}.0&"
            + f"ranking_period={rp_value}&sport_code={sport_id}"
        )
    elif sport_id == "WVB":
        url = (
            "https://stats.ncaa.org/rankings/institution_trends?"
            + f"academic_year={season+1}.0&division={ncaa_level}.0&"
            + f"ranking_period={rp_value}&sport_code={sport_id}"
        )

    best_method = True
    if (
        (season < 2017 and sport_id == "MVB")
    ):
        url = (
            "https://stats.ncaa.org/rankings/national_ranking?"
            + f"academic_year={season}.0&division={ncaa_level}.0&"
            + f"ranking_period={rp_value}&sport_code={sport_id}"
            + f"&stat_seq={stat_sequence}.0"
        )
        response = _get_webpage(url=url)
        best_method = False
    elif (
        (season < 2017 and sport_id == "WVB")
    ):
        url = (
            "https://stats.ncaa.org/rankings/national_ranking?"
            + f"academic_year={season+1}.0&division={ncaa_level}.0&"
            + f"ranking_period={rp_value}&sport_code={sport_id}"
            + f"&stat_seq={stat_sequence}.0"
        )
        response = _get_webpage(url=url)
        best_method = False
    elif sport_id == "MVB":
        try:
            response = _get_webpage(url=url)
        except Exception as e:
            logging.info(f"Found exception when loading teams `{e}`")
            logging.info("Attempting backup method.")
            url = (
                "https://stats.ncaa.org/rankings/national_ranking?"
                + f"academic_year={season}.0&division={ncaa_level}.0&"
                + f"ranking_period={rp_value}&sport_code={sport_id}"
                + f"&stat_seq={stat_sequence}.0"
            )
            response = _get_webpage(url=url)
            best_method = False
    else:
        try:
            response = _get_webpage(url=url)
        except Exception as e:
            logging.info(f"Found exception when loading teams `{e}`")
            logging.info("Attempting backup method.")
            url = (
                "https://stats.ncaa.org/rankings/national_ranking?"
                + f"academic_year={season+1}.0&division={ncaa_level}.0&"
                + f"ranking_period={rp_value}&sport_code={sport_id}"
                + f"&stat_seq={stat_sequence}.0"
            )
            response = _get_webpage(url=url)
            best_method = False

    soup = BeautifulSoup(response.text, features="lxml")

    if best_method is True:
        soup = soup.find(
            "table",
            {"id": "stat_grid"},
        )
        soup = soup.find("tbody")
        t_rows = soup.find_all("tr")

        for t in t_rows:
            team_id = t.find("a")
            team_id = team_id.get("href")
            team_id = team_id.replace("/teams/", "")
            team_id = int(team_id)
            team_name = t.find_all("td")[0].text
            team_conference_name = t.find_all("td")[1].text
            temp_df = pd.DataFrame(
                {
                    "season": season,
                    "ncaa_division": ncaa_level,
                    "ncaa_division_formatted": formatted_level,
                    "team_conference_name": team_conference_name,
                    "team_id": team_id,
                    "school_name": team_name,
                    "sport_id": sport_id,
                },
                index=[0],
            )
            teams_df_arr.append(temp_df)
            del temp_df
    else:
        soup = soup.find(
            "table",
            {"id": "rankings_table"},
        )
        soup = soup.find("tbody")
        t_rows = soup.find_all("tr")

        for t in t_rows:
            team_id = t.find("a")
            team_id = team_id.get("href")
            team_id = team_id.replace("/teams/", "")
            team_id = int(team_id)
            team = t.find_all("td")[1].get("data-order")
            team_name, team_conference_name = team.split(",")
            del team
            temp_df = pd.DataFrame(
                {
                    "season": season,
                    "ncaa_division": ncaa_level,
                    "ncaa_division_formatted": formatted_level,
                    "team_conference_name": team_conference_name,
                    "team_id": team_id,
                    "school_name": team_name,
                    "sport_id": sport_id,
                },
                index=[0],
            )
            teams_df_arr.append(temp_df)
            del temp_df

    teams_df = pd.concat(teams_df_arr, ignore_index=True)
    teams_df = pd.merge(
        left=teams_df,
        right=schools_df,
        on=["school_name"],
        how="left"
    )
    teams_df.sort_values(by=["team_id"], inplace=True)

    teams_df.to_csv(
        f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/teams/"
        + f"{season}_{formatted_level}_teams.csv",
        index=False,
    )

    return teams_df


def load_volleyball_teams(
    start_year: int = 2011,
    sport: str = None
) -> pd.DataFrame:
    """
    Compiles a list of known NCAA volleyball teams in NCAA volleyball history.

    Parameters
    ----------
    start_year : int, default 2011
        Specifies the first season you want NCAA volleyball team information from.

    sport : str, optional
        Sport type: "women"/"w"/"wvb" for women's volleyball (default),
        "men"/"m"/"mvb" for men's volleyball.
        If None, uses the configured default.

    Examples
    --------
    # Load in every women's volleyball team from 2011 to present day
    >>> df = load_volleyball_teams()
    
    # Load in every men's volleyball team from 2011 to present day  
    >>> df = load_volleyball_teams(sport="men")
    
    # Load in every women's volleyball team from 2020 to present day
    >>> df = load_volleyball_teams(start_year=2020)

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame object with a list of all known college volleyball teams.
    """
    if sport is None:
        sport = volleyball_config.default_sport
    
    get_mens_data, sport_id = _get_sport_params(sport)

    teams_df = pd.DataFrame()
    teams_df_arr = []
    temp_df = pd.DataFrame()

    now = datetime.now()
    mens_ncaa_divisions = ["I", "III"]
    womens_ncaa_divisions = ["I", "II", "III"]
    if now.month > 5 and get_mens_data is False:
        ncaa_seasons = [x for x in range(start_year, (now.year + 2))]
    elif now.month < 5 and get_mens_data is True:
        ncaa_seasons = [x for x in range(start_year, (now.year + 1))]
    else:
        ncaa_seasons = [x for x in range(start_year, (now.year + 1))]

    sport_name = "men's" if get_mens_data else "women's"
    logging.info(
        f"Loading in all NCAA {sport_name} volleyball teams. "
        + "If this is the first time you're seeing this message, "
        + "it may take some time (3-10 minutes) for this to load."
    )

    if get_mens_data is True:
        for s in ncaa_seasons:
            logging.info(
                f"Loading in men's volleyball teams for the {s} season."
            )
            for d in mens_ncaa_divisions:
                temp_df = _get_volleyball_teams_internal(
                    season=s,
                    level=d,
                    get_mens_data=True
                )
                teams_df_arr.append(temp_df)
                del temp_df
    else:
        for s in ncaa_seasons:
            logging.info(
                f"Loading in women's volleyball teams for the {s} season."
            )
            for d in womens_ncaa_divisions:
                temp_df = _get_volleyball_teams_internal(
                    season=s,
                    level=d,
                    get_mens_data=False
                )
                teams_df_arr.append(temp_df)
                del temp_df

    teams_df = pd.concat(teams_df_arr, ignore_index=True)
    teams_df = teams_df.infer_objects()
    return teams_df


def get_volleyball_team_schedule(team_id: int) -> pd.DataFrame:
    """
    Retrieves a team schedule, from a valid NCAA volleyball team ID.

    Parameters
    ----------
    team_id : int
        Required argument.
        Specifies the team you want a schedule from.
        This is separate from a school ID, which identifies the institution.
        A team ID should be unique to a school, and a season.

    Examples
    --------
    # Get the team schedule for the 2024 Toledo WVB team (D1, ID: 585329)
    >>> df = get_volleyball_team_schedule(585329)
    
    # Get the team schedule for the 2024 Hawaii MVB team (D1, ID: 573674)
    >>> df = get_volleyball_team_schedule(573674)

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame object with an NCAA volleyball team's schedule.
    """
    sport_id = ""
    schools_df = _get_schools()
    games_df = pd.DataFrame()
    games_df_arr = []
    season = 0
    temp_df = pd.DataFrame()
    load_from_cache = True

    home_dir = expanduser("~")
    home_dir = _format_folder_str(home_dir)

    url = f"https://stats.ncaa.org/teams/{team_id}"

    current_year = datetime.now().year
    estimated_season = current_year if datetime.now().month > 7 else current_year - 1

    season_found = False
    for year_offset in range(0, 3):
        try_season = estimated_season - year_offset
    
        # Try women's volleyball first
        for division in [1, 2, 3]:
            try:
                team_df = get_volleyball_teams(try_season, division, sport="women")
                team_match = team_df[team_df["team_id"] == team_id]
                
                if len(team_match) > 0:
                    season = team_match["season"].iloc[0]
                    ncaa_division = team_match["ncaa_division"].iloc[0]
                    ncaa_division_formatted = team_match["ncaa_division_formatted"].iloc[0]
                    sport_id = "WVB"
                    season_found = True
                    break
            except Exception:
                continue
        
        if season_found:
            break
            
        # Try men's volleyball
        for division in [1, 3]:
            try:
                team_df = get_volleyball_teams(try_season, division, sport="men")
                team_match = team_df[team_df["team_id"] == team_id]
                
                if len(team_match) > 0:
                    season = team_match["season"].iloc[0]
                    ncaa_division = team_match["ncaa_division"].iloc[0]
                    ncaa_division_formatted = team_match["ncaa_division_formatted"].iloc[0]
                    sport_id = "MVB"
                    season_found = True
                    break
            except Exception:
                continue
        
        if season_found:
            break

    del team_df

    if exists(f"{home_dir}/.ncaa_stats_py/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/")

    if exists(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/")

    if exists(
        f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/team_schedule/"
    ):
        pass
    else:
        mkdir(
            f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/team_schedule/"
        )

    if exists(
        f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/team_schedule/"
        + f"{team_id}_team_schedule.csv"
    ):
        games_df = pd.read_csv(
            f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/team_schedule/"
            + f"{team_id}_team_schedule.csv"
        )
        file_mod_datetime = datetime.fromtimestamp(
            getmtime(
                f"{home_dir}/.ncaa_stats_py/"
                + f"volleyball_{sport_id}/team_schedule/"
                + f"{team_id}_team_schedule.csv"
            )
        )
    else:
        file_mod_datetime = datetime.today()
        load_from_cache = False

    now = datetime.today()

    age = now - file_mod_datetime
    if (
        age.days > 1 and
        season >= now.year
    ):
        load_from_cache = False

    if load_from_cache is True:
        return games_df

    response = _get_webpage(url=url)
    soup = BeautifulSoup(response.text, features="lxml")

    school_name = soup.find("div", {"class": "card"}).find("img").get("alt")
    season_name = (
        soup.find("select", {"id": "year_list"})
        .find("option", {"selected": "selected"})
        .text
    )

    soup = soup.find_all(
        "div",
        {"class": "col p-0"},
    )

    table_data = ""
    for s in soup:
        try:
            temp_name = s.find("div", {"class": "card-header"})
            temp_name = temp_name.text
        except Exception as e:
            logging.warning(
                f"Could not parse card header. Full exception `{e}`. "
                + "Attempting alternate method."
            )
            temp_name = s.find("tr", {"class": "heading"}).find("td").text

        if "schedule" in temp_name.lower():
            table_data = s.find("table")

    t_rows = table_data.find_all("tr", {"class": "underline_rows"})

    if len(t_rows) == 0:
        t_rows = table_data.find_all("tr")

    for g in t_rows:
        is_valid_row = True
        game_num = 1
        ot_periods = 0
        is_home_game = True
        is_neutral_game = False

        cells = g.find_all("td")
        if len(cells) <= 1:
            continue

        game_date = cells[0].text

        if "(" in game_date:
            game_date = game_date.replace(")", "")
            game_date, game_num = game_date.split("(")
            game_date = game_date.strip()
            game_num = int(game_num.strip())

        if ":" in game_date and ("PM" in game_date or "AM" in game_date):
            game_date = datetime.strptime(
                game_date,
                "%m/%d/%Y %I:%M %p"
            ).date()
        else:
            game_date = datetime.strptime(
                game_date,
                "%m/%d/%Y"
            ).date()

        try:
            opp_team_id = cells[1].find("a").get("href")
        except IndexError:
            logging.info(
                "Skipping row because it is clearly "
                + "not a row that has schedule data."
            )
            is_valid_row = False
        except AttributeError as e:
            logging.info(
                "Could not extract a team ID for this game. " +
                f"Full exception {e}"
            )
            opp_team_id = "-1"
        except Exception as e:
            logging.warning(
                "An unhandled exception has occurred when "
                + "trying to get the opposition team ID for this game. "
                f"Full exception `{e}`."
            )
            raise e
        if is_valid_row is True:
            if opp_team_id is not None:
                opp_team_id = opp_team_id.replace("/teams/", "")
                opp_team_id = int(opp_team_id)

                try:
                    opp_team_name = cells[1].find("img").get("alt")
                except AttributeError:
                    logging.info(
                        "Couldn't find the opposition team name "
                        + "for this row from an image element. "
                        + "Attempting a backup method"
                    )
                    opp_team_name = cells[1].text
                except Exception as e:
                    logging.info(
                        "Unhandled exception when trying to get the "
                        + "opposition team name from this game. "
                        + f"Full exception `{e}`"
                    )
                    raise e
            else:
                opp_team_name = cells[1].text

            if opp_team_name[0] == "@":
                opp_team_name = opp_team_name.strip().replace("@", "")
            elif "@" in opp_team_name:
                opp_team_name = opp_team_name.strip().split("@")[0]

            opp_text = cells[1].text
            opp_text = opp_text.strip()
            if "@" in opp_text and opp_text[0] == "@":
                is_home_game = False
            elif "@" in opp_text and opp_text[0] != "@":
                is_neutral_game = True
                is_home_game = False
            # This is just to cover conference and NCAA championship
            # tournaments.
            elif "championship" in opp_text.lower():
                is_neutral_game = True
                is_home_game = False
            elif "ncaa" in opp_text.lower():
                is_neutral_game = True
                is_home_game = False

            del opp_text

            score = cells[2].text.strip()
            if len(score) == 0:
                score_1 = 0
                score_2 = 0
            elif (
                "canceled" not in score.lower() and
                "ppd" not in score.lower()
            ):
                score_1, score_2 = score.split("-")

                if any(x in score_1 for x in ["W", "L", "T"]):
                    score_1 = score_1.split(" ")[1]

                if "(" in score_2:
                    score_2 = score_2.replace(")", "")
                    score_2, ot_periods = score_2.split("(")
                    ot_periods = ot_periods.replace("OT", "")
                    ot_periods = ot_periods.replace(" ", "")
                    ot_periods = int(ot_periods)

                if ot_periods is None:
                    ot_periods = 0
                score_1 = int(score_1)
                score_2 = int(score_2)
            else:
                score_1 = None
                score_2 = None

            try:
                game_id = cells[2].find("a").get("href")
                game_id = game_id.replace("/contests", "")
                game_id = game_id.replace("/box_score", "")
                game_id = game_id.replace("/", "")
                game_id = int(game_id)
                game_url = (
                    f"https://stats.ncaa.org/contests/{game_id}/box_score"
                )
            except AttributeError as e:
                logging.info(
                    "Could not parse a game ID for this game. "
                    + f"Full exception `{e}`."
                )
                game_id = None
                game_url = None
            except Exception as e:
                logging.info(
                    "An unhandled exception occurred when trying "
                    + "to find a game ID for this game. "
                    + f"Full exception `{e}`."
                )
                raise e

            try:
                attendance = cells[3].text
                attendance = attendance.replace(",", "")
                attendance = attendance.replace("\n", "")
                attendance = int(attendance)
            except IndexError as e:
                logging.info(
                    "It doesn't appear as if there is an attendance column "
                    + "for this team's schedule table."
                    f"Full exception `{e}`."
                )
                attendance = None
            except ValueError as e:
                logging.info(
                    "There doesn't appear as if "
                    + "there is a recorded attendance. "
                    + "for this game/row. "
                    f"Full exception `{e}`."
                )
                attendance = None
            except Exception as e:
                logging.info(
                    "An unhandled exception occurred when trying "
                    + "to find this game's attendance. "
                    + f"Full exception `{e}`."
                )
                raise e

            if is_home_game is True:
                temp_df = pd.DataFrame(
                    {
                        "season": season,
                        "season_name": season_name,
                        "game_id": game_id,
                        "game_date": game_date,
                        "game_num": game_num,
                        "ot_periods": ot_periods,
                        "home_team_id": team_id,
                        "home_team_name": school_name,
                        "away_team_id": opp_team_id,
                        "away_team_name": opp_team_name,
                        "home_team_sets_won": score_1,
                        "away_team_sets_won": score_2,
                        "is_neutral_game": is_neutral_game,
                        "game_url": game_url,
                    },
                    index=[0],
                )
                games_df_arr.append(temp_df)
                del temp_df
            elif is_neutral_game is True:
                t_ids = [opp_team_id, team_id]
                t_ids.sort()

                if t_ids[0] == team_id:
                    temp_df = pd.DataFrame(
                        {
                            "season": season,
                            "season_name": season_name,
                            "game_id": game_id,
                            "game_date": game_date,
                            "game_num": game_num,
                            "ot_periods": ot_periods,
                            "home_team_id": team_id,
                            "home_team_name": school_name,
                            "away_team_id": opp_team_id,
                            "away_team_name": opp_team_name,
                            "home_team_sets_won": score_1,
                            "away_team_sets_won": score_2,
                            "is_neutral_game": is_neutral_game,
                            "game_url": game_url,
                        },
                        index=[0],
                    )

                else:
                    temp_df = pd.DataFrame(
                        {
                            "season": season,
                            "season_name": season_name,
                            "game_id": game_id,
                            "game_date": game_date,
                            "game_num": game_num,
                            "ot_periods": ot_periods,
                            "home_team_id": opp_team_id,
                            "home_team_name": opp_team_name,
                            "away_team_id": team_id,
                            "away_team_name": school_name,
                            "home_team_sets_won": score_2,
                            "away_team_sets_won": score_1,
                            "is_neutral_game": is_neutral_game,
                            "game_url": game_url,
                        },
                        index=[0],
                    )

                games_df_arr.append(temp_df)
                del temp_df
            else:
                temp_df = pd.DataFrame(
                    {
                        "season": season,
                        "season_name": season_name,
                        "game_id": game_id,
                        "game_date": game_date,
                        "game_num": game_num,
                        "ot_periods": ot_periods,
                        "home_team_id": opp_team_id,
                        "home_team_name": opp_team_name,
                        "away_team_id": team_id,
                        "away_team_name": school_name,
                        "home_team_sets_won": score_2,
                        "away_team_sets_won": score_1,
                        "is_neutral_game": is_neutral_game,
                        "game_url": game_url,
                    },
                    index=[0],
                )

                games_df_arr.append(temp_df)
                del temp_df

    games_df = pd.concat(games_df_arr, ignore_index=True)

    temp_df = schools_df.rename(
        columns={
            "school_name": "home_team_name",
            "school_id": "home_school_id"
        }
    )
    games_df = games_df.merge(right=temp_df, on="home_team_name", how="left")

    temp_df = schools_df.rename(
        columns={
            "school_name": "away_team_name",
            "school_id": "away_school_id"
        }
    )
    games_df = games_df.merge(right=temp_df, on="away_team_name", how="left")
    games_df["ncaa_division"] = ncaa_division
    games_df["ncaa_division_formatted"] = ncaa_division_formatted

    games_df.to_csv(
        f"{home_dir}/.ncaa_stats_py/"
        + f"volleyball_{sport_id}/team_schedule/"
        + f"{team_id}_team_schedule.csv",
        index=False,
    )

    return games_df


def get_volleyball_day_schedule(
    game_date: str | date | datetime,
    level: str | int = "I",
    sport: str = None
):
    """
    Given a date and NCAA level, this function retrieves volleyball every game
    for that date.

    Parameters
    ----------
    game_date : str | date | datetime
        Required argument.
        Specifies the date you want a volleyball schedule from.
        For best results, pass a string formatted as "YYYY-MM-DD".

    level : int | str, default "I"
        Required argument.
        Specifies the level/division you want a NCAA volleyball schedule from.
        This can either be an integer (1-3) or a string ("I"-"III").

    sport : str, optional
        Sport type: "women"/"w"/"wvb" for women's volleyball (default),
        "men"/"m"/"mvb" for men's volleyball.
        If None, uses the configured default.

    Examples
    --------
    # Get all DI women's games that were played on December 22th, 2024
    >>> df = get_volleyball_day_schedule("2024-12-22", level=1)
    
    # Get all division II women's games that were played on November 24th, 2024
    >>> df = get_volleyball_day_schedule("2024-11-24", level="II")
    
    # Get all DI men's games that will be played on April 12th, 2025
    >>> df = get_volleyball_day_schedule("2025-04-12", level=1, sport="men")

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame object with all volleyball games played on that day,
        for that NCAA division/level.
    """
    if sport is None:
        sport = volleyball_config.default_sport
    
    get_mens_data, sport_id = _get_sport_params(sport)
    
    season = 0

    schedule_df = pd.DataFrame()
    schedule_df_arr = []

    if isinstance(game_date, date):
        game_datetime = datetime.combine(
            game_date, datetime.min.time()
        )
    elif isinstance(game_date, datetime):
        game_datetime = game_date
    elif isinstance(game_date, str):
        game_datetime = parser.parse(
            game_date
        )
    else:
        unhandled_datatype = type(game_date)
        raise ValueError(
            f"Unhandled datatype for `game_date`: `{unhandled_datatype}`"
        )

    if isinstance(level, int) and level == 1:
        formatted_level = "I"
        ncaa_level = 1
    elif isinstance(level, int) and level == 2:
        formatted_level = "II"
        ncaa_level = 2
    elif isinstance(level, int) and level == 3:
        formatted_level = "III"
        ncaa_level = 3
    elif isinstance(level, str) and (
        level.lower() == "i" or level.lower() == "d1" or level.lower() == "1"
    ):
        ncaa_level = 1
        formatted_level = level.upper()
    elif isinstance(level, str) and (
        level.lower() == "ii" or level.lower() == "d2" or level.lower() == "2"
    ):
        ncaa_level = 2
        formatted_level = level.upper()
    elif isinstance(level, str) and (
        level.lower() == "iii" or level.lower() == "d3" or level.lower() == "3"
    ):
        ncaa_level = 3
        formatted_level = level.upper()

    del level

    season = game_datetime.year
    game_month = game_datetime.month
    game_day = game_datetime.day
    game_year = game_datetime.year

    if game_month > 7:
        season += 1
        url = (
            "https://stats.ncaa.org/contests/" +
            f"livestream_scoreboards?utf8=%E2%9C%93&sport_code={sport_id}" +
            f"&academic_year={season}&division={ncaa_level}" +
            f"&game_date={game_month:00d}%2F{game_day:00d}%2F{game_year}" +
            "&commit=Submit"
        )
    else:
        url = (
            "https://stats.ncaa.org/contests/" +
            f"livestream_scoreboards?utf8=%E2%9C%93&sport_code={sport_id}" +
            f"&academic_year={season}&division={ncaa_level}" +
            f"&game_date={game_month:00d}%2F{game_day:00d}%2F{game_year}" +
            "&commit=Submit"
        )

    response = _get_webpage(url=url)
    soup = BeautifulSoup(response.text, features="lxml")

    game_boxes = soup.find_all("div", {"class": "table-responsive"})

    for box in game_boxes:
        game_id = None
        game_alt_text = None
        game_num = 1
        table_box = box.find("table")
        table_rows = table_box.find_all("tr")

        # Date/attendance
        game_date_str = table_rows[0].find("div", {"class": "col-6 p-0"}).text
        game_date_str = game_date_str.replace("\n", "")
        game_date_str = game_date_str.strip()
        game_date_str = game_date_str.replace("TBA ", "TBA")
        game_date_str = game_date_str.replace("TBD ", "TBD")
        game_date_str = game_date_str.replace("PM ", "PM")
        game_date_str = game_date_str.replace("AM ", "AM")
        game_date_str = game_date_str.strip()
        attendance_str = table_rows[0].find(
            "div",
            {"class": "col p-0 text-right"}
        ).text

        attendance_str = attendance_str.replace("Attend:", "")
        attendance_str = attendance_str.replace(",", "")
        attendance_str = attendance_str.replace("\n", "")
        if (
            "st" in attendance_str.lower() or
            "nd" in attendance_str.lower() or
            "rd" in attendance_str.lower() or
            "th" in attendance_str.lower()
        ):
            attendance_num = None
        elif "final" in attendance_str.lower():
            attendance_num = None
        elif len(attendance_str) > 0:
            attendance_num = int(attendance_str)
        else:
            attendance_num = None

        if "(" in game_date_str:
            game_date_str = game_date_str.replace(")", "")
            game_date_str, game_num = game_date_str.split("(")
            game_num = int(game_num)

        if "TBA" in game_date_str:
            game_datetime = datetime.strptime(game_date_str, '%m/%d/%Y TBA')
        elif "tba" in game_date_str:
            game_datetime = datetime.strptime(game_date_str, '%m/%d/%Y tba')
        elif "TBD" in game_date_str:
            game_datetime = datetime.strptime(game_date_str, '%m/%d/%Y TBD')
        elif "tbd" in game_date_str:
            game_datetime = datetime.strptime(game_date_str, '%m/%d/%Y tbd')
        elif (
            "tbd" not in game_date_str.lower() and
            ":" not in game_date_str.lower()
        ):
            game_date_str = game_date_str.replace(" ", "")
            game_datetime = datetime.strptime(game_date_str, '%m/%d/%Y')
        else:
            game_datetime = datetime.strptime(
                game_date_str,
                '%m/%d/%Y %I:%M %p'
            )
        game_datetime = game_datetime.astimezone(timezone("US/Eastern"))

        game_alt_text = table_rows[1].find_all("td")[0].text
        if game_alt_text is not None and len(game_alt_text) > 0:
            game_alt_text = game_alt_text.replace("\n", "")
            game_alt_text = game_alt_text.strip()

        if len(game_alt_text) == 0:
            game_alt_text = None

        urls_arr = box.find_all("a")

        for u in urls_arr:
            url_temp = u.get("href")
            if "contests" in url_temp:
                game_id = url_temp
                del url_temp

        if game_id is None:
            for r in range(0, len(table_rows)):
                temp = table_rows[r]
                temp_id = temp.get("id")

                if temp_id is not None and len(temp_id) > 0:
                    game_id = temp_id

        del urls_arr

        game_id = game_id.replace("/contests", "")
        game_id = game_id.replace("/box_score", "")
        game_id = game_id.replace("/livestream_scoreboards", "")
        game_id = game_id.replace("/", "")
        game_id = game_id.replace("contest_", "")
        game_id = int(game_id)

        table_rows = table_box.find_all("tr", {"id": f"contest_{game_id}"})
        away_team_row = table_rows[0]
        home_team_row = table_rows[1]

        # Away team
        td_arr = away_team_row.find_all("td")

        try:
            away_team_name = td_arr[0].find("img").get("alt")
        except Exception:
            away_team_name = td_arr[1].text
        away_team_name = away_team_name.replace("\n", "")
        away_team_name = away_team_name.strip()

        try:
            away_team_id = td_arr[1].find("a").get("href")
            away_team_id = away_team_id.replace("/teams/", "")
            away_team_id = int(away_team_id)
        except AttributeError:
            away_team_id = None
            logging.info("No team ID found for the away team")
        except Exception as e:
            raise e

        away_sets_scored = td_arr[-1].text
        away_sets_scored = away_sets_scored.replace("\n", "")
        away_sets_scored = away_sets_scored.replace("\xa0", "")

        if "ppd" in away_sets_scored.lower():
            continue
        elif "cancel" in away_sets_scored.lower():
            continue

        if len(away_sets_scored) > 0:
            away_sets_scored = int(away_sets_scored)
        else:
            away_sets_scored = 0

        del td_arr

        # Home team
        td_arr = home_team_row.find_all("td")

        try:
            home_team_name = td_arr[0].find("img").get("alt")
        except Exception:
            home_team_name = td_arr[1].text
        home_team_name = home_team_name.replace("\n", "")
        home_team_name = home_team_name.strip()

        try:
            home_team_id = td_arr[1].find("a").get("href")
            home_team_id = home_team_id.replace("/teams/", "")
            home_team_id = int(home_team_id)
        except AttributeError:
            home_team_id = None
            logging.info("No team ID found for the home team")
        except Exception as e:
            raise e

        home_sets_scored = td_arr[-1].text
        home_sets_scored = home_sets_scored.replace("\n", "")
        home_sets_scored = home_sets_scored.replace("\xa0", "")

        if "ppd" in home_sets_scored.lower():
            continue
        elif "cancel" in home_sets_scored.lower():
            continue

        if len(home_sets_scored) > 0:
            home_sets_scored = int(home_sets_scored)
        else:
            home_sets_scored = 0

        temp_df = pd.DataFrame(
            {
                "season": season,
                "sport_id": sport_id,
                "game_date": game_datetime.strftime("%Y-%m-%d"),
                "game_datetime": game_datetime.isoformat(),
                "game_id": game_id,
                "formatted_level": formatted_level,
                "ncaa_level": ncaa_level,
                "game_alt_text": game_alt_text,
                "away_team_id": away_team_id,
                "away_team_name": away_team_name,
                "home_team_id": home_team_id,
                "home_team_name": home_team_name,
                "home_sets_scored": home_sets_scored,
                "away_sets_scored": away_sets_scored,
                "attendance": attendance_num
            },
            index=[0]
        )
        schedule_df_arr.append(temp_df)

        del temp_df

    if len(schedule_df_arr) >= 1:
        schedule_df = pd.concat(schedule_df_arr, ignore_index=True)
    else:
        logging.warning(
            "Could not find any game(s) for "
            + f"{game_datetime.year:00d}-{game_datetime.month:00d}"
            + f"-{game_datetime.day:00d}. "
            + "If you believe this is an error, "
            + "please raise an issue at "
            + "\n https://github.com/armstjc/ncaa_stats_py/issues \n"
        )
    return schedule_df


def get_full_volleyball_schedule(
    season: int,
    level: str | int = "I",
    sport: str = None
) -> pd.DataFrame:
    """
    Retrieves a full volleyball schedule from an NCAA level.
    The way this is done is by going through every team in a division,
    and parsing the schedules of every team in a division.

    This function will take time when first run (30-60 minutes)!
    You have been warned.

    Parameters
    ----------
    season : int
        Specifies the season you want a schedule from.

    level : int | str, default "I"
        Specifies the level/division you want a schedule from.

    sport : str, optional
        Sport type: "women"/"w"/"wvb" for women's volleyball (default),
        "men"/"m"/"mvb" for men's volleyball.
        If None, uses the configured default.

    Examples
    --------
    # Get the entire 2024 schedule for the 2024 women's D1 volleyball season
    >>> df = get_full_volleyball_schedule(season=2024, level="I")
    
    # Get the entire 2024 schedule for the 2024 men's D1 volleyball season
    >>> df = get_full_volleyball_schedule(season=2024, level="I", sport="men")

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame object with an NCAA volleyball
        schedule for a specific season and level.
    """
    if sport is None:
        sport = volleyball_config.default_sport
    
    get_mens_data, sport_id = _get_sport_params(sport)
    
    load_from_cache = True
    home_dir = expanduser("~")
    home_dir = _format_folder_str(home_dir)
    schedule_df = pd.DataFrame()
    schedule_df_arr = []
    temp_df = pd.DataFrame()
    formatted_level = ""
    ncaa_level = 0

    if isinstance(level, int) and level == 1:
        formatted_level = "I"
        ncaa_level = 1
    elif isinstance(level, int) and level == 2:
        formatted_level = "II"
        ncaa_level = 2
    elif isinstance(level, int) and level == 3:
        formatted_level = "III"
        ncaa_level = 3
    elif isinstance(level, str) and (
        level.lower() == "i" or level.lower() == "d1" or level.lower() == "1"
    ):
        ncaa_level = 1
        formatted_level = level.upper()
    elif isinstance(level, str) and (
        level.lower() == "ii" or level.lower() == "d2" or level.lower() == "2"
    ):
        ncaa_level = 2
        formatted_level = level.upper()
    elif isinstance(level, str) and (
        level.lower() == "iii" or level.lower() == "d3" or level.lower() == "3"
    ):
        ncaa_level = 3
        formatted_level = level.upper()

    del level

    if exists(f"{home_dir}/.ncaa_stats_py/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/")

    if exists(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/")

    if exists(
        f"{home_dir}/.ncaa_stats_py/" +
        f"volleyball_{sport_id}/full_schedule/"
    ):
        pass
    else:
        mkdir(
            f"{home_dir}/.ncaa_stats_py/" +
            f"volleyball_{sport_id}/full_schedule/"
        )

    if exists(
        f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/full_schedule/"
        + f"{season}_{formatted_level}_full_schedule.csv"
    ):
        teams_df = pd.read_csv(
            f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/full_schedule/"
            + f"{season}_{formatted_level}_full_schedule.csv"
        )
        file_mod_datetime = datetime.fromtimestamp(
            getmtime(
                f"{home_dir}/.ncaa_stats_py/" +
                f"volleyball_{sport_id}/full_schedule/"
                + f"{season}_{formatted_level}_full_schedule.csv"
            )
        )
    else:
        file_mod_datetime = datetime.today()
        load_from_cache = False

    now = datetime.today()

    age = now - file_mod_datetime

    if (
        age.days > 1 and
        season >= now.year
    ):
        load_from_cache = False

    if load_from_cache is True:
        return teams_df

    # FIXED: Use get_volleyball_teams() for specific season instead of load_volleyball_teams()
    # This prevents loading all teams from 2011 onwards
    teams_df = get_volleyball_teams(season=season, level=ncaa_level, sport=sport)
    team_ids_arr = teams_df["team_id"].to_numpy()

    for team_id in tqdm(team_ids_arr):
        temp_df = get_volleyball_team_schedule(team_id=team_id)
        schedule_df_arr.append(temp_df)

    schedule_df = pd.concat(schedule_df_arr, ignore_index=True)
    schedule_df = schedule_df.drop_duplicates(subset="game_id", keep="first")
    schedule_df.to_csv(
        f"{home_dir}/.ncaa_stats_py/"
        + f"volleyball_{sport_id}/full_schedule/"
        + f"{season}_{formatted_level}_full_schedule.csv",
        index=False,
    )
    return schedule_df

def get_volleyball_team_roster(team_id: int) -> pd.DataFrame:
    """
    Retrieves a volleyball team's roster from a given team ID.

    Parameters
    ----------
    team_id : int
        Required argument.
        Specifies the team you want a roster from.
        This is separate from a school ID, which identifies the institution.
        A team ID should be unique to a school, and a season.

    Examples
    --------
    # Get the volleyball roster for the 2024 Weber St. WVB team (D1, ID: 585347)
    >>> df = get_volleyball_team_roster(585347)
    
    # Get the volleyball roster for the 2024 Hawaii MVB team (D1, ID: 573674)
    >>> df = get_volleyball_team_roster(573674)

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame object with an NCAA volleyball team's roster for that season.
    """
    sport_id = ""
    roster_df = pd.DataFrame()
    roster_df_arr = []
    temp_df = pd.DataFrame()
    url = f"https://stats.ncaa.org/teams/{team_id}/roster"
    load_from_cache = True
    home_dir = expanduser("~")
    home_dir = _format_folder_str(home_dir)

    stat_columns = [
        "season",
        "season_name",
        "sport_id",
        "ncaa_division",
        "ncaa_division_formatted",
        "team_conference_name",
        "school_id",
        "school_name",
        "player_id",
        "player_jersey_num",
        "player_full_name",
        "player_first_name",
        "player_last_name",
        "player_class",
        "player_positions",
        "player_height_string",
        "player_weight",
        "player_hometown",
        "player_high_school",
        "player_G",
        "player_GS",
        "player_url",
    ]

    current_year = datetime.now().year
    estimated_season = current_year if datetime.now().month > 7 else current_year - 1

    season_found = False
    for year_offset in range(0, 3):
        try_season = estimated_season - year_offset
    
        # Try women's volleyball first
        for division in [1, 2, 3]:
            try:
                team_df = get_volleyball_teams(try_season, division, sport="women")
                team_match = team_df[team_df["team_id"] == team_id]
                
                if len(team_match) > 0:
                    season = team_match["season"].iloc[0]
                    ncaa_division = team_match["ncaa_division"].iloc[0]
                    ncaa_division_formatted = team_match["ncaa_division_formatted"].iloc[0]
                    sport_id = "WVB"
                    season_found = True
                    break
            except Exception:
                continue
        
        if season_found:
            break
            
        # Try men's volleyball
        for division in [1, 3]:
            try:
                team_df = get_volleyball_teams(try_season, division, sport="men")
                team_match = team_df[team_df["team_id"] == team_id]
                
                if len(team_match) > 0:
                    season = team_match["season"].iloc[0]
                    ncaa_division = team_match["ncaa_division"].iloc[0]
                    ncaa_division_formatted = team_match["ncaa_division_formatted"].iloc[0]
                    sport_id = "MVB"
                    season_found = True
                    break
            except Exception:
                continue
        
        if season_found:
            break

    del team_df

    if exists(f"{home_dir}/.ncaa_stats_py/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/")

    if exists(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/")

    if exists(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/rosters/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/rosters/")

    if exists(
        f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/rosters/" +
        f"{team_id}_roster.csv"
    ):
        teams_df = pd.read_csv(
            f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/rosters/" +
            f"{team_id}_roster.csv"
        )
        file_mod_datetime = datetime.fromtimestamp(
            getmtime(
                f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/rosters/" +
                f"{team_id}_roster.csv"
            )
        )
    else:
        file_mod_datetime = datetime.today()
        load_from_cache = False

    now = datetime.today()

    age = now - file_mod_datetime

    if (
        age.days >= 14 and
        season >= now.year
    ):
        load_from_cache = False

    if load_from_cache is True:
        return teams_df

    response = _get_webpage(url=url)
    soup = BeautifulSoup(response.text, features="lxml")
    try:
        school_name = soup.find(
            "div",
            {"class": "card"}
        ).find("img").get("alt")
    except Exception:
        school_name = soup.find("div", {"class": "card"}).find("a").text
        school_name = school_name.rsplit(" ", maxsplit=1)[0]

    season_name = (
        soup.find("select", {"id": "year_list"})
        .find("option", {"selected": "selected"})
        .text
    )

    try:
        table = soup.find(
            "table",
            {"class": "dataTable small_font"},
        )

        table_headers = table.find("thead").find_all("th")
    except Exception:
        table = soup.find(
            "table",
            {"class": "dataTable small_font no_padding"},
        )

        table_headers = table.find("thead").find_all("th")
    table_headers = [x.text for x in table_headers]

    t_rows = table.find("tbody").find_all("tr")

    for t in t_rows:
        t_cells = t.find_all("td")
        t_cells = [x.text for x in t_cells]

        temp_df = pd.DataFrame(
            data=[t_cells],
            columns=table_headers,
        )

        player_id = t.find("a").get("href")
        temp_df["player_url"] = f"https://stats.ncaa.org{player_id}"

        player_id = player_id.replace("/players", "").replace("/", "")
        player_id = int(player_id)

        temp_df["player_id"] = player_id

        roster_df_arr.append(temp_df)
        del temp_df

    roster_df = pd.concat(roster_df_arr, ignore_index=True)
    roster_df = roster_df.infer_objects()
    roster_df["season"] = season
    roster_df["season_name"] = season_name
    roster_df["ncaa_division"] = ncaa_division
    roster_df["ncaa_division_formatted"] = ncaa_division_formatted
    roster_df["team_conference_name"] = team_conference_name
    roster_df["school_id"] = school_id
    roster_df["school_name"] = school_name
    roster_df["sport_id"] = sport_id

    roster_df.rename(
        columns={
            "GP": "player_G",
            "GS": "player_GS",
            "#": "player_jersey_num",
            "Name": "player_full_name",
            "Class": "player_class",
            "Position": "player_positions",
            "Height": "player_height_string",
            "Bats": "player_batting_hand",
            "Throws": "player_throwing_hand",
            "Hometown": "player_hometown",
            "High School": "player_high_school",
        },
        inplace=True
    )

    roster_df[["player_first_name", "player_last_name"]] = roster_df[
        "player_full_name"
    ].str.split(" ", n=1, expand=True)
    roster_df = roster_df.infer_objects()

    for i in roster_df.columns:
        if i in stat_columns:
            pass
        else:
            raise ValueError(
                f"Unhandled column name {i}"
            )

    roster_df = roster_df.infer_objects().reindex(columns=stat_columns)

    roster_df.to_csv(
        f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/rosters/" +
        f"{team_id}_roster.csv",
        index=False,
    )
    return roster_df


def get_volleyball_player_season_stats(team_id: int, team_info: dict = None) -> pd.DataFrame:
    """
    Given a team ID, this function retrieves and parses
    the season stats for all of the players in a given volleyball team.

    Parameters
    ----------
    team_id : int
        Required argument.
        Specifies the team you want volleyball stats from.
        This is separate from a school ID, which identifies the institution.
        A team ID should be unique to a school, and a season.
    
    team_info : dict, optional
        Team metadata to avoid loading all teams data. Should contain:
        season, ncaa_division, ncaa_division_formatted, team_conference_name,
        school_name, school_id, sport_id

    Examples
    --------
    # Get the season stats for the 2024 Ohio St. team (D1, ID: 585398)
    >>> df = get_volleyball_player_season_stats(585398)
    
    # Get the season stats with team info to avoid data loading
    >>> team_info = {...}  # from get_volleyball_teams()
    >>> df = get_volleyball_player_season_stats(585398, team_info)

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame object with the season stats for
        all players with a given NCAA volleyball team.
    """
    sport_id = ""
    load_from_cache = True
    stats_df = pd.DataFrame()
    stats_df_arr = []
    temp_df = pd.DataFrame()

    stat_columns = [
        "season",
        "season_name",
        "sport_id",
        "team_id",
        "team_conference_name",
        "school_id",
        "school_name",
        "ncaa_division",
        "ncaa_division_formatted",
        "player_id",
        "player_jersey_number",
        "player_last_name",
        "player_first_name",
        "player_full_name",
        "player_class",
        "player_position",
        "player_height",
        "GP",
        "GS",
        "sets_played",
        "MS",
        "kills",
        "errors",
        "total_attacks",
        "hit%",
        "assists",
        "aces",
        "serve_errors",
        "digs",
        "return_attacks",
        "return_errors",
        "solo_blocks",
        "assisted_blocks",
        "block_errors",
        "total_blocks",
        "points",
        "BHE",
        "serve_attempts",
        "DBL_DBL",
        "TRP_DBL",
    ]

    # Use provided team_info if available to avoid loading all teams data
    if team_info is not None:
        season = team_info["season"]
        ncaa_division = team_info["ncaa_division"]
        ncaa_division_formatted = team_info["ncaa_division_formatted"]
        team_conference_name = team_info["team_conference_name"]
        school_name = team_info["school_name"]
        school_id = int(team_info["school_id"])
        sport_id = team_info["sport_id"]
    else:
        # Fallback to original method - try to find team efficiently
        current_year = datetime.now().year
        estimated_season = current_year if datetime.now().month > 7 else current_year - 1
        
        season_found = False
        for year_offset in range(0, 5):  # Check current and previous 4 years
            try_season = estimated_season - year_offset
            
            try:
                # Load only specific season data
                team_df = get_volleyball_teams(try_season, "I")
                team_match = team_df[team_df["team_id"] == team_id]
                
                if len(team_match) > 0:
                    season = team_match["season"].iloc[0]
                    ncaa_division = team_match["ncaa_division"].iloc[0]
                    ncaa_division_formatted = team_match["ncaa_division_formatted"].iloc[0]
                    team_conference_name = team_match["team_conference_name"].iloc[0]
                    school_name = team_match["school_name"].iloc[0]
                    school_id = int(team_match["school_id"].iloc[0])
                    sport_id = "WVB"
                    season_found = True
                    break
            except Exception:
                continue
        
        # Try other divisions if not found in D1
        if not season_found:
            for division in [2, 3]:
                for year_offset in range(0, 5):
                    try_season = estimated_season - year_offset
                    try:
                        team_df = get_volleyball_teams(try_season, division)
                        team_match = team_df[team_df["team_id"] == team_id]
                        
                        if len(team_match) > 0:
                            season = team_match["season"].iloc[0]
                            ncaa_division = team_match["ncaa_division"].iloc[0]
                            ncaa_division_formatted = team_match["ncaa_division_formatted"].iloc[0]
                            team_conference_name = team_match["team_conference_name"].iloc[0]
                            school_name = team_match["school_name"].iloc[0]
                            school_id = int(team_match["school_id"].iloc[0])
                            sport_id = "WVB"
                            season_found = True
                            break
                    except Exception:
                        continue
                if season_found:
                    break
        
        # Try men's volleyball if still not found
        if not season_found:
            for division in [1, 3]:  # Men's volleyball only has D1 and D3
                for year_offset in range(0, 5):
                    try_season = estimated_season - year_offset
                    try:
                        team_df = get_volleyball_teams(try_season, division, sport="men")
                        team_match = team_df[team_df["team_id"] == team_id]
                        
                        if len(team_match) > 0:
                            season = team_match["season"].iloc[0]
                            ncaa_division = team_match["ncaa_division"].iloc[0]
                            ncaa_division_formatted = team_match["ncaa_division_formatted"].iloc[0]
                            team_conference_name = team_match["team_conference_name"].iloc[0]
                            school_name = team_match["school_name"].iloc[0]
                            school_id = int(team_match["school_id"].iloc[0])
                            sport_id = "MVB"
                            season_found = True
                            break
                    except Exception:
                        continue
                if season_found:
                    break

        # Final fallback to original inefficient method
        if not season_found:
            try:
                team_df = load_volleyball_teams()
                team_df = team_df[team_df["team_id"] == team_id]

                season = team_df["season"].iloc[0]
                ncaa_division = team_df["ncaa_division"].iloc[0]
                ncaa_division_formatted = team_df["ncaa_division_formatted"].iloc[0]
                team_conference_name = team_df["team_conference_name"].iloc[0]
                school_name = team_df["school_name"].iloc[0]
                school_id = int(team_df["school_id"].iloc[0])
                sport_id = "WVB"
            except Exception:
                team_df = load_volleyball_teams(sport="men")
                team_df = team_df[team_df["team_id"] == team_id]

                season = team_df["season"].iloc[0]
                ncaa_division = team_df["ncaa_division"].iloc[0]
                ncaa_division_formatted = team_df["ncaa_division_formatted"].iloc[0]
                team_conference_name = team_df["team_conference_name"].iloc[0]
                school_name = team_df["school_name"].iloc[0]
                school_id = int(team_df["school_id"].iloc[0])
                sport_id = "MVB"

    home_dir = expanduser("~")
    home_dir = _format_folder_str(home_dir)

    url = f"https://stats.ncaa.org/teams/{team_id}/season_to_date_stats"

    if exists(f"{home_dir}/.ncaa_stats_py/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/")

    if exists(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/"):
        pass
    else:
        mkdir(f"{home_dir}/.ncaa_stats_py/volleyball_{sport_id}/")

    if exists(
        f"{home_dir}/.ncaa_stats_py/" +
        f"volleyball_{sport_id}/player_season_stats/"
    ):
        pass
    else:
        mkdir(
            f"{home_dir}/.ncaa_stats_py/" +
            f"volleyball_{sport_id}/player_season_stats/"
        )

    if exists(
        f"{home_dir}/.ncaa_stats_py/" +
        f"volleyball_{sport_id}/player_season_stats/"
        + f"{season:00d}_{school_id:00d}_player_season_stats.csv"
    ):
        games_df = pd.read_csv(
            f"{home_dir}/.ncaa_stats_py/" +
            f"volleyball_{sport_id}/player_season_stats/"
            + f"{season:00d}_{school_id:00d}_player_season_stats.csv"
        )
        file_mod_datetime = datetime.fromtimestamp(
            getmtime(
                f"{home_dir}/.ncaa_stats_py/" +
                f"volleyball_{sport_id}/player_season_stats/"
                + f"{season:00d}_{school_id:00d}_player_season_stats.csv"
            )
        )
    else:
        file_mod_datetime = datetime.today()
        load_from_cache = False

    now = datetime.today()

    age = now - file_mod_datetime

    if (
        age.days > 1 and
        season >= now.year
    ):
        load_from_cache = False

    if load_from_cache is True:
        return games_df

    response = _get_webpage(url=url)

    soup = BeautifulSoup(response.text, features="lxml")

    # Check if we got a valid page
    season_select = soup.find("select", {"id": "year_list"})
    if season_select is None:
        logging.warning(f"Could not find season selector for team {team_id}. Page may not have stats data.")
        return pd.DataFrame()  # Return empty DataFrame

    season_name = (
        season_select
        .find("option", {"selected": "selected"})
        .text
    )

    if sport_id == "MVB":
        season = f"{season_name[0:2]}{season_name[-2:]}"
        season = int(season)
    elif sport_id == "WVB":
        season = f"{season_name[0:4]}"
        season = int(season)

    table_data = soup.find(
        "table",
        {"id": "stat_grid", "class": "small_font dataTable table-bordered"},
    )

    # Check if stats table exists
    if table_data is None:
        logging.warning(f"Could not find stats table for team {team_id}. Team may not have player stats available.")
        return pd.DataFrame()  # Return empty DataFrame

    thead = table_data.find("thead")
    if thead is None:
        logging.warning(f"Could not find table header for team {team_id}.")
        return pd.DataFrame()

    temp_table_headers = thead.find("tr").find_all("th")
    table_headers = [x.text for x in temp_table_headers]

    del temp_table_headers

    t_rows = table_data.find("tbody").find_all("tr", {"class": "text"})
    for t in t_rows:
        p_last = ""
        p_first = ""
        t_cells = t.find_all("td")
        if "team" in t_cells[1].text.lower():
            continue
        p_sortable = t_cells[1].get("data-order")
        if len(p_sortable) == 2:
            p_last, p_first = p_sortable.split(",")
        elif len(p_sortable) == 3:
            p_last, temp_name, p_first = p_sortable.split(",")
            p_last = f"{p_last} {temp_name}"

        t_cells = [x.text.strip() for x in t_cells]
        t_cells = [x.replace(",", "") for x in t_cells]

        temp_df = pd.DataFrame(
            data=[t_cells],
            columns=table_headers,
        )

        player_id = t.find("a").get("href")

        player_id = player_id.replace("/players", "").replace("/", "")

        player_id = int(player_id)

        temp_df["player_id"] = player_id
        temp_df["player_last_name"] = p_last.strip()
        temp_df["player_first_name"] = p_first.strip()

        stats_df_arr.append(temp_df)
        del temp_df

    stats_df = pd.concat(stats_df_arr, ignore_index=True)
    stats_df = stats_df.replace("", None)

    stats_df["season"] = season
    stats_df["season_name"] = season_name
    stats_df["school_id"] = school_id
    stats_df["school_name"] = school_name
    stats_df["ncaa_division"] = ncaa_division
    stats_df["ncaa_division_formatted"] = ncaa_division_formatted
    stats_df["team_conference_name"] = team_conference_name
    stats_df["sport_id"] = sport_id
    stats_df["team_id"] = team_id

    stats_df = stats_df.infer_objects()

    stats_df.rename(
        columns={
            "#": "player_jersey_number",
            "Player": "player_full_name",
            "Yr": "player_class",
            "Pos": "player_position",
            "Ht": "player_height",
            "S": "sets_played",
            "Kills": "kills",
            "Errors": "errors",
            "Total Attacks": "total_attacks",
            "Hit Pct": "hit%",
            "Assists": "assists",
            "Aces": "aces",
            "SErr": "serve_errors",
            "Digs": "digs",
            "RetAtt": "return_attacks",
            "RErr": "return_errors",
            "Block Solos": "solo_blocks",
            "Block Assists": "assisted_blocks",
            "BErr": "block_errors",
            "PTS": "points",
            "Trpl Dbl": "TRP_DBL",
            "Dbl Dbl": "DBL_DBL",
            "TB": "total_blocks",
            "SrvAtt": "serve_attempts",
        },
        inplace=True,
    )

    for i in stats_df.columns:
        if i in stat_columns:
            pass
        elif "Attend" in stat_columns:
            pass
        else:
            raise ValueError(
                f"Unhandled column name {i}"
            )
    stats_df = stats_df.reindex(columns=stat_columns)

    stats_df = stats_df.infer_objects().fillna(0)
    stats_df = stats_df.astype(
        {
            "GP": "uint16",
            "GS": "uint16",
            "sets_played": "uint16",
            "kills": "uint16",
            "errors": "uint16",
            "total_attacks": "uint16",
            "hit%": "float32",
            "assists": "uint16",
            "aces": "uint16",
            "serve_errors": "uint16",
            "digs": "uint16",
            "return_attacks": "uint16",
            "return_errors": "uint16",
            "solo_blocks": "uint16",
            "assisted_blocks": "uint16",
            "block_errors": "uint16",
            "points": "float32",
            "BHE": "uint16",
            "TRP_DBL": "uint16",
            "serve_attempts": "uint16",
            "total_blocks": "float32",
            "DBL_DBL": "uint16",
            "school_id": "uint32",
        }
    )

    stats_df["hit%"] = stats_df["hit%"].round(3)
    stats_df["points"] = stats_df["points"].round(1)

    stats_df.to_csv(
        f"{home_dir}/.ncaa_stats_py/" +
        f"volleyball_{sport_id}/player_season_stats/" +
        f"{season:00d}_{school_id:00d}_player_season_stats.csv",
        index=False,
    )

    return stats_df

def summarize_volleyball_season(season: int, level: str | int = 1, sport: str = None) -> dict:
    """
    Get a quick summary of a volleyball season.
    
    Parameters
    ----------
    season : int
        The season year
    level : str | int, default 1
        NCAA division
    sport : str, optional
        Sport type, if None uses configured default
        
    Examples
    --------
    >>> summary = summarize_volleyball_season(2024, 1, "women")
    >>> print(f"Found {summary['total_teams']} teams in {summary['conferences']} conferences")
    
    Returns
    -------
    dict
        Summary information about the volleyball season
    """
    if sport is None:
        sport = volleyball_config.default_sport
        
    teams_df = get_volleyball_teams(season, level, sport)
    get_mens_data, sport_id = _get_sport_params(sport)
    
    summary = {
        'season': season,
        'sport': sport,
        'level': level,
        'total_teams': len(teams_df),
        'conferences': teams_df['team_conference_name'].nunique(),
        'conference_list': sorted(teams_df['team_conference_name'].unique()),
        'data_cache_location': f"{expanduser('~')}/.ncaa_stats_py/volleyball_{sport_id}/"
    }
    
    return summary


def get_womens_volleyball_season_data(season: int, level: str | int = 1) -> dict:
    """
    Get complete women's volleyball season data in one call.
    
    Parameters
    ----------
    season : int
        The season year
    level : str | int, default 1
        NCAA division
        
    Examples
    --------
    >>> data = get_womens_volleyball_season_data(2024, 1)
    >>> teams = data['teams']
    >>> schedule = data['schedule'] 
    >>> player_stats = data['player_stats']
    
    Returns
    -------
    dict
        Dictionary with keys: 'teams', 'schedule', 'player_stats'
    """
    teams_df = get_womens_volleyball_teams(season, level)
    
    logging.info("Loading full schedule...")
    schedule_df = get_full_volleyball_schedule(season, level, sport="women")
    
    # Get player stats for all teams
    logging.info("Loading player stats for all teams...")
    player_stats_list = []
    for _, team_row in tqdm(teams_df.iterrows(), total=len(teams_df), desc="Loading player stats"):
        try:
            # Pass team info to avoid lookup
            team_info = {
                "season": team_row["season"],
                "ncaa_division": team_row["ncaa_division"],
                "ncaa_division_formatted": team_row["ncaa_division_formatted"],
                "team_conference_name": team_row["team_conference_name"],
                "school_name": team_row["school_name"],
                "school_id": team_row["school_id"],
                "sport_id": "WVB"
            }
            stats = get_volleyball_player_season_stats(team_row['team_id'], team_info)
            player_stats_list.append(stats)
        except Exception as e:
            logging.warning(f"Failed to get stats for team {team_row['team_id']}: {e}")
    
    player_stats_df = pd.concat(player_stats_list, ignore_index=True) if player_stats_list else pd.DataFrame()
    
    return {
        'teams': teams_df,
        'schedule': schedule_df, 
        'player_stats': player_stats_df
    }


def get_mens_volleyball_season_data(season: int, level: str | int = 1) -> dict:
    """
    Get complete men's volleyball season data in one call.
    
    Parameters
    ----------
    season : int
        The season year
    level : str | int, default 1
        NCAA division (1 or 3, no D2 men's volleyball)
        
    Examples
    --------
    >>> data = get_mens_volleyball_season_data(2024, 1)
    >>> teams = data['teams']
    >>> schedule = data['schedule'] 
    >>> player_stats = data['player_stats']
    
    Returns
    -------
    dict
        Dictionary with keys: 'teams', 'schedule', 'player_stats'
    """
    teams_df = get_mens_volleyball_teams(season, level)
    
    logging.info("Loading full schedule...")
    schedule_df = get_full_volleyball_schedule(season, level, sport="men")
    
    # Get player stats for all teams
    logging.info("Loading player stats for all teams...")
    player_stats_list = []
    for _, team_row in tqdm(teams_df.iterrows(), total=len(teams_df), desc="Loading player stats"):
        try:
            # Pass team info to avoid lookup
            team_info = {
                "season": team_row["season"],
                "ncaa_division": team_row["ncaa_division"],
                "ncaa_division_formatted": team_row["ncaa_division_formatted"],
                "team_conference_name": team_row["team_conference_name"],
                "school_name": team_row["school_name"],
                "school_id": team_row["school_id"],
                "sport_id": "MVB"
            }
            stats = get_volleyball_player_season_stats(team_row['team_id'], team_info)
            player_stats_list.append(stats)
        except Exception as e:
            logging.warning(f"Failed to get stats for team {team_row['team_id']}: {e}")
    
    player_stats_df = pd.concat(player_stats_list, ignore_index=True) if player_stats_list else pd.DataFrame()
    
    return {
        'teams': teams_df,
        'schedule': schedule_df, 
        'player_stats': player_stats_df
    }