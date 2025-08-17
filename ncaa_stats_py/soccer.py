# Author: Derek Willis (dwillis@gmail.com)
# File Name: `soccer.py`
# Purpose: Houses functions that allows one to access NCAA soccer data
# Creation Date: 2024-09-20 08:15 PM EDT


import logging
import re
from datetime import date, datetime
from os import mkdir
from os.path import exists, expanduser, getmtime

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from ncaa_stats_py.utls import (
    _format_folder_str,
    _get_schools,
    _get_seconds_from_time_str,
    _get_stat_id,
    _get_webpage,
)


def get_soccer_teams(
    season: int, level: str | int,
    get_womens_soccer_data: bool = False
) -> pd.DataFrame:
    """
    Retrieves a list of soccer teams from the NCAA.

    Parameters
    ----------
    `season` (int, mandatory): season year
    `level` (int or str, mandatory): division (1/2/3 or "I"/"II"/"III")
    `get_womens_soccer_data` (bool, optional): toggles women's data

    Notes
    -----
    This function mirrors the style of `lacrosse.get_lacrosse_teams`.
    Uses `WSO` and `MSO` for team sport codes per repository convention.
    The cache directory is `.ncaa_stats_py/soccer_{sport_id}`.
    """
    sport_id = "WSO" if get_womens_soccer_data else "MSO"
    load_from_cache = True
    home_dir = expanduser("~")
    home_dir = _format_folder_str(home_dir)
    teams_df = pd.DataFrame()
    teams_df_arr = []
    temp_df = pd.DataFrame()
    formatted_level = ""
    ncaa_level = 0

    # Set sport codes and keys for stat lookup
    if get_womens_soccer_data is True:
        sport_code = "WSO"
        sport_key = "womens_soccer"
    else:
        sport_code = "MSO"
        sport_key = "mens_soccer"

    # For soccer: user season (2025) means Fall 2025 = Academic Year 2025-26
    stat_dict_season = season

    try:
        stat_sequence = _get_stat_id(sport_key, stat_dict_season, "team")
    except LookupError:
        logging.warning(f"Could not find team stat ID for {sport_key} season {stat_dict_season}")
        # Fallback to hardcoded values
        if get_womens_soccer_data:
            stat_sequence = 56
        else:
            stat_sequence = 30

    # Normalize level input
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
        level.lower() in {"i", "d1", "1"}
    ):
        formatted_level = "I"
        ncaa_level = 1
    elif isinstance(level, str) and (
        level.lower() in {"ii", "d2", "2"}
    ):
        formatted_level = "II"
        ncaa_level = 2
    elif isinstance(level, str) and (
        level.lower() in {"iii", "d3", "3"}
    ):
        formatted_level = "III"
        ncaa_level = 3
    else:
        raise ValueError("Invalid 'level' parameter for get_soccer_teams")

    # Ensure cache directories exist (using os.makedirs for robustness)
    import os
    base_cache_dir = f"{home_dir}/.ncaa_stats_py"
    soccer_cache_dir = f"{base_cache_dir}/soccer_{sport_id}"
    teams_cache_dir = f"{soccer_cache_dir}/teams"
    for d in [base_cache_dir, soccer_cache_dir, teams_cache_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    cache_file = f"{teams_cache_dir}/{season}_{formatted_level}_teams.csv"

    if exists(cache_file):
        teams_df = pd.read_csv(cache_file)
        file_mod_datetime = datetime.fromtimestamp(getmtime(cache_file))
    else:
        file_mod_datetime = datetime.today()
        load_from_cache = False

    now = datetime.today()
    age = now - file_mod_datetime

    if (
        age.days > 1 and
        season >= now.year and
        now.month <= 7
    ):
        load_from_cache = False
    elif (
        age.days >= 14 and
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
    # Deduplicate schools_df on school_name before merging
    schools_df = schools_df.drop_duplicates(subset=["school_name"]).copy()
    
    change_url = (
        "https://stats.ncaa.org/rankings/change_sport_year_div?"
        + f"academic_year={season + 1}.0&division={ncaa_level}.0"
        + f"&sport_code={sport_code}"
    )

    inst_url = (
        "https://stats.ncaa.org/rankings/institution_trends?"
        + f"academic_year={season + 1}.0&division={ncaa_level}.0&"
        + f"ranking_period=0&sport_code={sport_code}"
        + f"&stat_seq={stat_sequence}"
    )

    # Use the shared _get_webpage utility for all requests (like lacrosse.py)
    inst_html = _get_webpage(inst_url)
    soup = BeautifulSoup(inst_html.text, features="lxml")

    # Try parsing common table layouts used on stats.ncaa.org
    try:
        # Prefer extracting team rows by finding anchors that link to /teams/<id>
        # This is more robust than relying on column positions since layouts vary.
        anchors = soup.find_all('a', href=True)
        found = {}
        for a in anchors:
            href = a.get('href')
            if not href or '/teams/' not in href:
                continue
            # Heuristics to ensure this anchor is a team link shown in rankings
            is_team_anchor = False
            # target="TEAM_WIN" is used on the rankings table
            if a.get('target') and a.get('target').upper() == 'TEAM_WIN':
                is_team_anchor = True
            # class 'skipMask' appears on team anchors
            if not is_team_anchor:
                cls = a.get('class') or []
                if isinstance(cls, list) and 'skipMask' in cls:
                    is_team_anchor = True
            # extract team id
            try:
                team_id = int(href.split('/teams/')[1].split('/')[0])
            except Exception:
                team_id = None
            # find row and parent table
            tr = a.find_parent('tr')
            if tr is None:
                continue
            table = tr.find_parent('table')
            # check parent td classes (e.g., 'sorting_1')
            if not is_team_anchor:
                parent_td = a.find_parent('td')
                if parent_td is not None:
                    pcls = parent_td.get('class') or []
                    if isinstance(pcls, list) and any('sorting' in c for c in pcls):
                        is_team_anchor = True
            if not is_team_anchor:
                continue
            # ensure this table looks like a rankings table (has Team/School header or known id)
            is_rankings = False
            if table is not None:
                tid = table.get('id')
                if tid in ('stat_grid', 'rankings_table'):
                    is_rankings = True
                else:
                    thead = table.find('thead')
                    if thead:
                        head_txt = ' '.join([th.get_text(' ', strip=True).lower() for th in thead.find_all('th')])
                        if any(k in head_txt for k in ('team', 'school', 'institution')):
                            is_rankings = True
            # skip anchors that are not in a likely rankings table
            if not is_rankings:
                continue
            # extract name and conference heuristics
            team_name = a.get_text(strip=True)
            team_conf = ''
            tds = tr.find_all('td')
            if tds:
                # prefer data-order attr if present
                for cell in tds:
                    data_order = cell.get('data-order')
                    if data_order and ',' in data_order:
                        parts = [p.strip() for p in data_order.split(',', 1)]
                        if len(parts) == 2:
                            team_name = parts[0]
                            team_conf = parts[1]
                            break
                # fallback conference in next td if available
                if not team_conf and len(tds) > 1:
                    team_conf = tds[1].get_text(strip=True)
            # normalize and dedupe
            if team_name:
                key = (team_id, team_name)
                if key in found:
                    continue
                found[key] = True
                temp_df = pd.DataFrame({
                    "season": season,
                    "ncaa_division": ncaa_level,
                    "ncaa_division_formatted": formatted_level,
                    "team_conference_name": team_conf,
                    "team_id": team_id,
                    "school_name": team_name,
                    "sport_id": sport_id,
                }, index=[0])
                teams_df_arr.append(temp_df)

        # If anchor-based extraction found rows, skip the older heuristics
        if not teams_df_arr:
            # fallback: previous table-based heuristics (keep as-is)
            table = soup.find("table", {"id": "stat_grid"})
            if table is not None:
                tbody = table.find("tbody")
                t_rows = tbody.find_all("tr") if tbody is not None else []
                for t in t_rows:
                    # Attempt several heuristics to extract team id, name, conference
                    team_id = None
                    team_name = None
                    team_conf = ""
                    # 1) look for anchor with /teams/ID
                    try:
                        a = t.find('a', href=True)
                        if a and '/teams/' in a['href']:
                            try:
                                team_id = int(a['href'].split('/teams/')[1].split('/')[0])
                            except Exception:
                                team_id = None
                            team_name = a.get_text(strip=True) or None
                    except Exception:
                        pass
                    # 2) some tables embed team,conference in data-order attr on a td
                    if not team_name:
                        try:
                            td = t.find_all('td')
                            for cell in td:
                                data_order = cell.get('data-order')
                                if data_order and ',' in data_order:
                                    team_name, team_conf = [x.strip() for x in data_order.split(',', 1)]
                                    break
                            # fallback: visible text in first or second td
                            if not team_name and td:
                                if len(td) > 1:
                                    team_name = td[1].get_text(strip=True)
                                else:
                                    team_name = td[0].get_text(strip=True)
                        except Exception:
                            pass
                    # 3) if team_name is numeric-like (a stat), skip this row
                    if team_name:
                        try:
                            float(team_name.replace(',', ''))
                            # numeric-only; skip
                            continue
                        except Exception:
                            # not numeric, ok
                            pass
                    # final guard: must have a team_name
                    if not team_name:
                        continue
                    # normalize
                    team_name = team_name.strip()
                    temp_df = pd.DataFrame(
                        {
                            "season": season,
                            "ncaa_division": ncaa_level,
                            "ncaa_division_formatted": formatted_level,
                            "team_conference_name": team_conf,
                            "team_id": team_id,
                            "school_name": team_name,
                            "sport_id": sport_id,
                        },
                        index=[0],
                    )
                    teams_df_arr.append(temp_df)
            else:
                # Try the rankings_table layout
                table = soup.find("table", {"id": "rankings_table"})
                if table is not None:
                    tbody = table.find("tbody")
                    t_rows = tbody.find_all("tr") if tbody is not None else []
                    for t in t_rows:
                        try:
                            team_link = t.find("a")
                            team_id = int(team_link.get("href").replace("/teams/", ""))
                        except Exception:
                            continue
                        # some rows embed team,conference in data-order attribute
                        team = t.find_all("td")[1].get("data-order")
                        if team and "," in team:
                            team_name, team_conf = team.split(",", 1)
                        else:
                            # fallback: visible text
                            team_name = t.find_all("td")[1].text.strip()
                            team_conf = ""
                        temp_df = pd.DataFrame(
                            {
                                "season": season,
                                "ncaa_division": ncaa_level,
                                "ncaa_division_formatted": formatted_level,
                                "team_conference_name": team_conf,
                                "team_id": team_id,
                                "school_name": team_name,
                                "sport_id": sport_id,
                            },
                            index=[0],
                        )
                        teams_df_arr.append(temp_df)
                else:
                    # Generic table parse: first <table> on page
                    table = soup.find("table")
                    if table is not None:
                        rows = table.find_all("tr")
                        for r in rows:
                            cols = r.find_all("td")
                            if not cols:
                                continue
                            # try extract link and name
                            try:
                                team_link = r.find("a")
                                team_id = int(team_link.get("href").replace("/teams/", ""))
                            except Exception:
                                team_id = None
                            school_name = cols[1].get_text(strip=True) if len(cols) > 1 else cols[0].get_text(strip=True)
                            temp_df = pd.DataFrame(
                                {
                                    "season": season,
                                    "ncaa_division": ncaa_level,
                                    "ncaa_division_formatted": formatted_level,
                                    "team_conference_name": "",
                                    "team_id": team_id,
                                    "school_name": school_name,
                                    "sport_id": sport_id,
                                },
                                index=[0],
                            )
                            teams_df_arr.append(temp_df)
    except Exception as e:
        logging.warning(f"Failed to parse soccer teams: {e}")

    if not teams_df_arr:
        return pd.DataFrame()

    teams_df = pd.concat(teams_df_arr, ignore_index=True)
    teams_df = pd.merge(
        left=teams_df,
        right=schools_df,
        on=["school_name"],
        how="left",
    )
    # Deduplicate merged teams_df on season and team_id
    teams_df = teams_df.drop_duplicates(subset=["season", "team_id"]).copy()
    teams_df.sort_values(by=["team_id"], inplace=True)

    # Ensure the teams cache directory exists before writing
    teams_cache_dir = os.path.dirname(cache_file)
    if not os.path.exists(teams_cache_dir):
        os.makedirs(teams_cache_dir)
    teams_df.to_csv(cache_file, index=False)

    return teams_df


def load_soccer_teams(
    start_year: int = 2010,
    get_womens_soccer_data: bool = False
) -> pd.DataFrame:
    """
    Compiles a list of known NCAA soccer teams from `start_year` to present.
    """
    if get_womens_soccer_data is True:
        sport_id = "WSO"
    else:
        sport_id = "MSO"

    teams_df_arr = []
    now = datetime.now()
    ncaa_seasons = [x for x in range(start_year, (now.year))]

    logging.info("Loading soccer teams across seasons; this may take a while")
    for s in tqdm(ncaa_seasons, desc="seasons"):
        for div in ["I", "II", "III"]:
            try:
                df = get_soccer_teams(s, div, get_womens_soccer_data=get_womens_soccer_data)
                if df is not None and not df.empty:
                    df["season"] = s
                    df["ncaa_division"] = div
                    teams_df_arr.append(df)
            except Exception:
                logging.debug(f"Skipping season {s} division {div} due to error")

    if teams_df_arr:
        teams_df = pd.concat(teams_df_arr, ignore_index=True)
        teams_df = teams_df.infer_objects()
        return teams_df

    return pd.DataFrame()

def get_soccer_team_schedule(
    team_id: int, 
    season: int, 
    get_womens_soccer_data: bool = False
) -> pd.DataFrame:
    """
    Retrieves a team schedule, from a valid NCAA soccer team ID.
    
    Parameters
    ----------
    team_id : int
        The NCAA team ID
    season : int
        The season year (e.g., 2025 for Fall 2025 season)
    get_womens_soccer_data : bool, optional
        Whether to get women's soccer data (default: False for men's)
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the team's schedule
    """
    
    # Set sport_id based on parameter
    sport_id = "WSO" if get_womens_soccer_data else "MSO"
    
    schools_df = _get_schools()
    games_df = pd.DataFrame()
    games_df_arr = []
    temp_df = pd.DataFrame()
    load_from_cache = True

    home_dir = expanduser("~")
    home_dir = _format_folder_str(home_dir)

    # Try to get team info from teams cache first
    team_info_found = False
    ncaa_division = 1
    ncaa_division_formatted = "I"
    school_name = None
    
    try:
        # Load teams for the specified sport and find the team
        teams_df = get_soccer_teams(season, "I", get_womens_soccer_data=get_womens_soccer_data)
        if not teams_df.empty:
            team_row = teams_df[teams_df["team_id"] == team_id]
            if not team_row.empty:
                ncaa_division = team_row["ncaa_division"].iloc[0]
                ncaa_division_formatted = team_row["ncaa_division_formatted"].iloc[0]
                school_name = team_row["school_name"].iloc[0]
                team_info_found = True
                logging.info(f"Found team {team_id} in Division {ncaa_division_formatted} teams cache")
        
        # If not found in Division I, try other divisions
        if not team_info_found:
            for div in ["II", "III"]:
                teams_df = get_soccer_teams(season, div, get_womens_soccer_data=get_womens_soccer_data)
                if not teams_df.empty:
                    team_row = teams_df[teams_df["team_id"] == team_id]
                    if not team_row.empty:
                        ncaa_division = team_row["ncaa_division"].iloc[0]
                        ncaa_division_formatted = team_row["ncaa_division_formatted"].iloc[0]
                        school_name = team_row["school_name"].iloc[0]
                        team_info_found = True
                        logging.info(f"Found team {team_id} in Division {ncaa_division_formatted} teams cache")
                        break
                        
    except Exception as e:
        logging.warning(f"Could not find team in teams cache: {e}")

    # If team not found in cache, extract info from the team page directly
    if not team_info_found:
        logging.warning(f"Team ID {team_id} not found in teams cache. Extracting info from team page.")
        try:
            url = f"https://stats.ncaa.org/teams/{team_id}"
            response = _get_webpage(url=url)
            soup = BeautifulSoup(response.text, features="lxml")
            
            # Extract school name from the page
            try:
                school_name = soup.find("div", {"class": "card"}).find("img").get("alt")
            except:
                # Fallback method
                try:
                    school_name = soup.find("title").text.split(" - ")[0]
                except:
                    school_name = f"Team {team_id}"
            
            logging.info(f"Extracted school name: {school_name}")
            
        except Exception as e:
            logging.error(f"Could not extract team info from page: {e}")
            school_name = f"Team {team_id}"

    # Get stat sequence for the sport and season
    try:
        sport_key = "womens_soccer" if get_womens_soccer_data else "mens_soccer"
        stat_sequence = _get_stat_id(sport_key, season, "team")
        logging.info(f"Found stat ID {stat_sequence} for {sport_key} season {season}")
    except LookupError:
        logging.warning(f"Could not find team stat ID for {sport_key} season {season}")
        # Use fallback values
        stat_sequence = 56 if get_womens_soccer_data else 30

    # Ensure cache directories exist
    import os
    base_cache_dir = f"{home_dir}/.ncaa_stats_py"
    soccer_cache_dir = f"{base_cache_dir}/soccer_{sport_id}"
    schedule_cache_dir = f"{soccer_cache_dir}/team_schedule"
    
    for d in [base_cache_dir, soccer_cache_dir, schedule_cache_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    cache_file = f"{schedule_cache_dir}/{team_id}_team_schedule.csv"

    # Check if we should load from cache
    if os.path.exists(cache_file):
        games_df = pd.read_csv(cache_file)
        file_mod_datetime = datetime.fromtimestamp(os.path.getmtime(cache_file))
    else:
        file_mod_datetime = datetime.today()
        load_from_cache = False

    now = datetime.today()
    age = now - file_mod_datetime
    
    # Determine if cache is stale
    if age.days > 1 and season >= now.year and now.month <= 7:
        load_from_cache = False

    if load_from_cache:
        return games_df

    # Scrape the schedule from the team page
    url = f"https://stats.ncaa.org/teams/{team_id}"
    response = _get_webpage(url=url)
    soup = BeautifulSoup(response.text, features="lxml")

    # Get school name if we don't have it yet
    if not school_name:
        try:
            school_name = soup.find("div", {"class": "card"}).find("img").get("alt")
        except:
            school_name = f"Team {team_id}"

    # Get season name
    try:
        season_name = (
            soup.find("select", {"id": "year_list"})
            .find("option", {"selected": "selected"})
            .text
        )
    except:
        season_name = f"{season}-{str(season+1)[-2:]}"

    # Find the schedule table
    soup_sections = soup.find_all("div", {"class": "col p-0"})
    
    table_data = None
    for s in soup_sections:
        try:
            temp_name = s.find("div", {"class": "card-header"})
            temp_name = temp_name.text
        except Exception as e:
            logging.warning(f"Could not parse card header. Full exception `{e}`. Attempting alternate method.")
            try:
                temp_name = s.find("tr", {"class": "heading"}).find("td").text
            except:
                continue

        if "schedule" in temp_name.lower():
            table_data = s.find("table")
            break

    if table_data is None:
        logging.warning("Could not find schedule table on page")
        return pd.DataFrame()

    # Parse schedule rows
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

        # Parse game date
        game_date = cells[0].text.strip()
        if not game_date or game_date.lower() in ['date', 'game']:
            continue

        # Handle game series notation
        if "(" in game_date:
            game_date = game_date.replace(")", "")
            game_date, game_num = game_date.split("(")
            game_date = game_date.strip()
            game_num = int(game_num.strip())

        try:
            game_date = datetime.strptime(game_date, "%m/%d/%Y").date()
        except ValueError:
            continue  # Skip invalid date rows

        # Parse opponent
        try:
            opp_team_link = cells[1].find("a")
            if opp_team_link:
                opp_team_id = opp_team_link.get("href")
                if opp_team_id:
                    opp_team_id = opp_team_id.replace("/teams/", "")
                    opp_team_id = int(opp_team_id)
                else:
                    opp_team_id = None
            else:
                opp_team_id = None
        except (IndexError, AttributeError, ValueError):
            logging.info("Skipping row because it doesn't contain valid team data.")
            continue
        except Exception as e:
            logging.warning(f"Unhandled exception when parsing opponent team ID: {e}")
            opp_team_id = None

        # Parse opponent name
        try:
            opp_team_name = cells[1].find("img").get("alt")
        except AttributeError:
            opp_team_name = cells[1].text.strip()
        except Exception as e:
            logging.warning(f"Error parsing opponent name: {e}")
            continue

        # Determine home/away/neutral
        opp_text = cells[1].text.strip()
        if "@" in opp_text and opp_text.startswith("@"):
            is_home_game = False
            opp_team_name = opp_team_name.replace("@", "").strip()
        elif "@" in opp_text and not opp_text.startswith("@"):
            is_neutral_game = True
            is_home_game = False
            opp_team_name = opp_text.split("@")[0].strip()
        elif any(keyword in opp_text.lower() for keyword in ["championship", "ncaa", "tournament"]):
            is_neutral_game = True
            is_home_game = False

        # Parse score
        score = cells[2].text.strip()
        if len(score) == 0 or any(word in score.lower() for word in ["canceled", "ppd", "postponed"]):
            score_1 = None
            score_2 = None
        else:
            try:
                # Handle overtime notation
                score = score.replace("(-3 OT)", "")
                
                # Remove win/loss indicators
                if any(x in score for x in ["W", "L", "T"]):
                    parts = score.split(" ")
                    if len(parts) >= 2:
                        score = parts[1]

                score_1, score_2 = score.split("-")
                
                # Handle overtime periods
                if "(" in score_2:
                    score_2 = score_2.replace(")", "")
                    score_2, ot_text = score_2.split("(")
                    ot_periods = int(re.findall(r'\d+', ot_text)[0]) if re.findall(r'\d+', ot_text) else 0

                score_1 = int(score_1.strip())
                score_2 = int(score_2.strip())
            except (ValueError, IndexError):
                score_1 = None
                score_2 = None

        # Parse game ID and URL
        try:
            game_link = cells[2].find("a")
            if game_link:
                game_href = game_link.get("href")
                game_id = int(re.findall(r'/contests/(\d+)', game_href)[0])
                game_url = f"https://stats.ncaa.org/contests/{game_id}/box_score"
            else:
                game_id = None
                game_url = None
        except (AttributeError, IndexError, ValueError):
            game_id = None
            game_url = None

        # Parse attendance
        try:
            if len(cells) > 3:
                attendance = cells[3].text.strip().replace(",", "").replace("\n", "")
                attendance = int(attendance) if attendance.isdigit() else None
            else:
                attendance = None
        except (ValueError, IndexError):
            attendance = None

        # Create game record
        if is_home_game:
            temp_df = pd.DataFrame({
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
                "home_team_score": score_1,
                "away_team_score": score_2,
                "is_neutral_game": is_neutral_game,
                "game_url": game_url,
                "attendance": attendance,
            }, index=[0])
        elif is_neutral_game:
            # Order team IDs for consistent neutral game representation
            if opp_team_id and team_id:
                t_ids = [opp_team_id, team_id]
                t_ids.sort()
                
                if t_ids[0] == team_id:
                    temp_df = pd.DataFrame({
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
                        "home_team_score": score_1,
                        "away_team_score": score_2,
                        "is_neutral_game": is_neutral_game,
                        "game_url": game_url,
                        "attendance": attendance,
                    }, index=[0])
                else:
                    temp_df = pd.DataFrame({
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
                        "home_team_score": score_2,
                        "away_team_score": score_1,
                        "is_neutral_game": is_neutral_game,
                        "game_url": game_url,
                        "attendance": attendance,
                    }, index=[0])
            else:
                # Fallback if opp_team_id is None
                temp_df = pd.DataFrame({
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
                    "home_team_score": score_1,
                    "away_team_score": score_2,
                    "is_neutral_game": is_neutral_game,
                    "game_url": game_url,
                    "attendance": attendance,
                }, index=[0])
        else:  # Away game
            temp_df = pd.DataFrame({
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
                "home_team_score": score_2,
                "away_team_score": score_1,
                "is_neutral_game": is_neutral_game,
                "game_url": game_url,
                "attendance": attendance,
            }, index=[0])

        games_df_arr.append(temp_df)

    if not games_df_arr:
        return pd.DataFrame()

    # Combine all games
    games_df = pd.concat(games_df_arr, ignore_index=True)

    # Merge with school information
    temp_df = schools_df.rename(columns={
        "school_name": "home_team_name",
        "school_id": "home_school_id"
    })
    games_df = games_df.merge(right=temp_df, on="home_team_name", how="left")

    temp_df = schools_df.rename(columns={
        "school_name": "away_team_name", 
        "school_id": "away_school_id"
    })
    games_df = games_df.merge(right=temp_df, on="away_team_name", how="left")
    
    # Add division information
    games_df["ncaa_division"] = ncaa_division
    games_df["ncaa_division_formatted"] = ncaa_division_formatted

    # Save to cache
    games_df.to_csv(cache_file, index=False)

    return games_df

def get_soccer_day_schedule(
    game_date: str | date | datetime,
    level: str | int = "I",
    get_womens_soccer_data: bool = False
) -> pd.DataFrame:
    """
    Given a date and NCAA level, this function retrieves every soccer game
    for that date.

    Parameters
    ----------
    `game_date` (str | date | datetime, mandatory):
        Required argument.
        Specifies the date you want a soccer schedule from.
        For best results, pass a string formatted as "YYYY-MM-DD".

    `level` (str | int, optional):
        Optional argument (default: "I").
        Specifies the level/division you want a
        NCAA soccer schedule from.
        This can either be an integer (1-3) or a string ("I"-"III").

    `get_womens_soccer_data` (bool, optional):
        Optional argument (default: False).
        If you want women's soccer data instead of men's soccer data,
        set this to `True`.

    Usage
    ----------
    ```python

    from ncaa_stats_py.soccer import get_soccer_day_schedule

    ###########################################
    #              Men's soccer               #
    ###########################################

    # Get all DI games that will be played on September 8th, 2025.
    print("Get all games that will be played on September 8th, 2025.")
    df = get_soccer_day_schedule("2025-09-08", level=1)
    print(df)

    # Get all division III games will be played on October 15th, 2025.
    print("Get all division III games will be played on October 15th, 2025.")
    df = get_soccer_day_schedule("2025-10-15", level="III")
    print(df)

    # Get all DI games that were played on November 27th, 2024.
    print("Get all games that were played on November 27th, 2024.")
    df = get_soccer_day_schedule("2024-11-27", level="I")
    print(df)

    # Get all DI games (if any) that were played on September 23rd, 2024.
    print("Get all DI games (if any) that were played on September 23rd, 2024.")
    df = get_soccer_day_schedule("2024-09-23")
    print(df)

    # Get all DIII games played on October 9th, 2024.
    print("Get all DIII games played on October 9th, 2024.")
    df = get_soccer_day_schedule("2024-10-09", level="III")
    print(df)

    ###########################################
    #             Women's soccer              #
    ###########################################

    # Get all DI games that will be played on September 8th, 2025.
    print("Get all games that will be played on September 8th, 2025.")
    df = get_soccer_day_schedule(
        "2025-09-08", level=1, get_womens_soccer_data=True
    )
    print(df)

    # Get all division III games will be played on October 15th, 2025.
    print("Get all division III games will be played on October 15th, 2025.")
    df = get_soccer_day_schedule(
        "2025-10-15", level="III", get_womens_soccer_data=True
    )
    print(df)

    # Get all DI games that were played on November 27th, 2024.
    print("Get all games that were played on November 27th, 2024.")
    df = get_soccer_day_schedule(
        "2024-11-27", level="I", get_womens_soccer_data=True
    )
    print(df)

    ```

    Returns
    ----------
    A pandas `DataFrame` object with all soccer games played on that day,
    for that NCAA division/level.

    """
    from dateutil import parser
    from pytz import timezone
    
    season = 0
    sport_id = "MSO"

    schedule_df = pd.DataFrame()
    schedule_df_arr = []

    # Parse the input date
    if isinstance(game_date, date):
        game_datetime = datetime.combine(
            game_date, datetime.min.time()
        )
    elif isinstance(game_date, datetime):
        game_datetime = game_date
    elif isinstance(game_date, str):
        game_datetime = parser.parse(game_date)
    else:
        unhandled_datatype = type(game_date)
        raise ValueError(
            f"Unhandled datatype for `game_date`: `{unhandled_datatype}`"
        )

    # Parse the level parameter
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
        level.lower() in {"i", "d1", "1"}
    ):
        ncaa_level = 1
        formatted_level = "I"
    elif isinstance(level, str) and (
        level.lower() in {"ii", "d2", "2"}
    ):
        ncaa_level = 2
        formatted_level = "II"
    elif isinstance(level, str) and (
        level.lower() in {"iii", "d3", "3"}
    ):
        ncaa_level = 3
        formatted_level = "III"
    else:
        raise ValueError(f"Invalid 'level' parameter: {level}")

    # Set sport ID based on gender
    if get_womens_soccer_data is True:
        sport_id = "WSO"
    elif get_womens_soccer_data is False:
        sport_id = "MSO"
    else:
        raise ValueError(
            f"Unhandled value for `get_womens_soccer_data`: `{get_womens_soccer_data}`"
        )

    # Determine season based on date
    # Soccer season runs in fall, so academic year calculation differs from spring sports
    game_month = game_datetime.month
    game_day = game_datetime.day
    game_year = game_datetime.year

    # For soccer: August-December games are in academic year starting that year
    # January-July games are in academic year that started the previous year
    if game_month >= 8:  # August through December
        season = game_year + 1  # Academic year 2025-26 for fall 2025 games
    else:  # January through July
        season = game_year  # Academic year 2025-26 for spring 2025 games

    # Build URL for NCAA stats scoreboard
    url = (
        "https://stats.ncaa.org/contests/" +
        f"livestream_scoreboards?utf8=%E2%9C%93&sport_code={sport_id}" +
        f"&academic_year={season}&division={ncaa_level}" +
        f"&game_date={game_month:02d}%2F{game_day:02d}%2F{game_year}" +
        "&commit=Submit"
    )

    response = _get_webpage(url=url)
    soup = BeautifulSoup(response.text, features="lxml")

    game_boxes = soup.find_all("div", {"class": "table-responsive"})

    for box in game_boxes:
        game_id = None
        game_alt_text = None
        game_num = 1
        ot_periods = 0
        
        table_box = box.find("table")
        if table_box is None:
            continue
            
        table_rows = table_box.find_all("tr")
        if len(table_rows) < 2:
            continue

        # Parse date/attendance from first row
        try:
            game_date_str = table_rows[0].find("div", {"class": "col-6 p-0"}).text
            game_date_str = game_date_str.replace("\n", "").strip()
            game_date_str = game_date_str.replace("TBA ", "TBA")
            game_date_str = game_date_str.replace("TBD ", "TBD")
            game_date_str = game_date_str.replace("PM ", "PM")
            game_date_str = game_date_str.replace("AM ", "AM")
            game_date_str = game_date_str.strip()
        except (AttributeError, IndexError):
            continue

        try:
            attendance_str = table_rows[0].find(
                "div", {"class": "col p-0 text-right"}
            ).text
            attendance_str = attendance_str.replace("Attend:", "")
            attendance_str = attendance_str.replace(",", "")
            attendance_str = attendance_str.replace("\n", "").strip()
            
            if (
                any(x in attendance_str.lower() for x in ["st", "nd", "rd", "th"]) or
                "final" in attendance_str.lower() or
                len(attendance_str) == 0
            ):
                attendance_num = None
            else:
                try:
                    attendance_num = int(attendance_str)
                except ValueError:
                    attendance_num = None
        except (AttributeError, IndexError):
            attendance_num = None

        # Handle game series notation
        if "(" in game_date_str:
            game_date_str = game_date_str.replace(")", "")
            game_date_str, game_num = game_date_str.split("(")
            game_num = int(game_num)

        # Parse game datetime
        try:
            if "TBA" in game_date_str or "TBD" in game_date_str:
                game_datetime_parsed = datetime.strptime(
                    game_date_str.split()[0], '%m/%d/%Y'
                )
            elif ":" not in game_date_str:
                game_date_str = game_date_str.replace(" ", "")
                game_datetime_parsed = datetime.strptime(game_date_str, '%m/%d/%Y')
            else:
                game_datetime_parsed = datetime.strptime(
                    game_date_str, '%m/%d/%Y %I:%M %p'
                )
            
            # Convert to Eastern timezone
            game_datetime_parsed = game_datetime_parsed.replace(
                tzinfo=timezone("US/Eastern")
            )
        except ValueError:
            continue

        # Get game alternative text/description
        try:
            game_alt_text = table_rows[1].find_all("td")[0].text
            if game_alt_text is not None and len(game_alt_text) > 0:
                game_alt_text = game_alt_text.replace("\n", "").strip()
            if len(game_alt_text) == 0:
                game_alt_text = None
        except (AttributeError, IndexError):
            game_alt_text = None

        # Find game ID from links or row IDs
        urls_arr = box.find_all("a")
        for u in urls_arr:
            url_temp = u.get("href")
            if url_temp and "contests" in url_temp:
                game_id = url_temp
                break

        if game_id is None:
            for r in range(len(table_rows)):
                temp = table_rows[r]
                temp_id = temp.get("id")
                if temp_id is not None and len(temp_id) > 0:
                    game_id = temp_id
                    break

        if game_id is None:
            continue

        # Clean up game ID
        game_id = game_id.replace("/contests", "")
        game_id = game_id.replace("/box_score", "")
        game_id = game_id.replace("/livestream_scoreboards", "")
        game_id = game_id.replace("/", "")
        game_id = game_id.replace("contest_", "")
        try:
            game_id = int(game_id)
        except ValueError:
            continue

        # Find team rows
        team_rows = table_box.find_all("tr", {"id": f"contest_{game_id}"})
        if len(team_rows) < 2:
            # Try alternative method - look for rows with team data
            team_rows = []
            for row in table_rows:
                if row.find("td") and len(row.find_all("td")) >= 3:
                    team_rows.append(row)
            
            if len(team_rows) < 2:
                continue
                
        away_team_row = team_rows[0]
        home_team_row = team_rows[1]

        # Parse away team information
        try:
            away_td_arr = away_team_row.find_all("td")
            if len(away_td_arr) < 2:
                continue

            # Get away team name
            try:
                away_team_name = away_td_arr[0].find("img").get("alt")
            except (AttributeError, IndexError):
                away_team_name = away_td_arr[1].text if len(away_td_arr) > 1 else away_td_arr[0].text
            away_team_name = away_team_name.replace("\n", "").strip()

            # Get away team ID
            try:
                away_team_link = away_td_arr[1].find("a")
                if away_team_link:
                    away_team_id = away_team_link.get("href")
                    away_team_id = away_team_id.replace("/teams/", "")
                    away_team_id = int(away_team_id)
                else:
                    away_team_id = None
            except (AttributeError, ValueError, IndexError):
                away_team_id = None

            # Get away team score
            away_score = away_td_arr[-1].text.replace("\n", "").replace("\xa0", "").strip()
            
            if any(word in away_score.lower() for word in ["canceled", "ppd", "postponed"]):
                continue
            
            # Handle overtime notation in scores
            if "(" in away_score and "OT" in away_score:
                score_parts = away_score.split("(")
                away_score = score_parts[0].strip()
                ot_text = score_parts[1].replace(")", "").strip()
                ot_periods = int(re.findall(r'\d+', ot_text)[0]) if re.findall(r'\d+', ot_text) else 1

            try:
                away_goals_scored = int(away_score) if away_score else 0
            except ValueError:
                away_goals_scored = 0

        except (AttributeError, IndexError):
            continue

        # Parse home team information
        try:
            home_td_arr = home_team_row.find_all("td")
            if len(home_td_arr) < 2:
                continue

            # Get home team name
            try:
                home_team_name = home_td_arr[0].find("img").get("alt")
            except (AttributeError, IndexError):
                home_team_name = home_td_arr[1].text if len(home_td_arr) > 1 else home_td_arr[0].text
            home_team_name = home_team_name.replace("\n", "").strip()

            # Get home team ID
            try:
                home_team_link = home_td_arr[1].find("a")
                if home_team_link:
                    home_team_id = home_team_link.get("href")
                    home_team_id = home_team_id.replace("/teams/", "")
                    home_team_id = int(home_team_id)
                else:
                    home_team_id = None
            except (AttributeError, ValueError, IndexError):
                home_team_id = None

            # Get home team score
            home_score = home_td_arr[-1].text.replace("\n", "").replace("\xa0", "").strip()
            
            # Handle overtime notation in scores (if not already handled)
            if "(" in home_score and "OT" in home_score and ot_periods == 0:
                score_parts = home_score.split("(")
                home_score = score_parts[0].strip()
                ot_text = score_parts[1].replace(")", "").strip()
                ot_periods = int(re.findall(r'\d+', ot_text)[0]) if re.findall(r'\d+', ot_text) else 1

            try:
                home_goals_scored = int(home_score) if home_score else 0
            except ValueError:
                home_goals_scored = 0

        except (AttributeError, IndexError):
            continue

        # Create DataFrame row for this game
        temp_df = pd.DataFrame({
            "season": season,
            "sport_id": sport_id,
            "game_date": game_datetime_parsed.strftime("%Y-%m-%d"),
            "game_datetime": game_datetime_parsed.isoformat(),
            "game_id": game_id,
            "game_num": game_num,
            "ot_periods": ot_periods,
            "formatted_level": formatted_level,
            "ncaa_level": ncaa_level,
            "game_alt_text": game_alt_text,
            "away_team_id": away_team_id,
            "away_team_name": away_team_name,
            "home_team_id": home_team_id,
            "home_team_name": home_team_name,
            "home_goals_scored": home_goals_scored,
            "away_goals_scored": away_goals_scored,
            "attendance": attendance_num
        }, index=[0])
        
        schedule_df_arr.append(temp_df)

    # Combine all games into a single DataFrame
    if len(schedule_df_arr) >= 1:
        schedule_df = pd.concat(schedule_df_arr, ignore_index=True)
    else:
        logging.warning(
            f"Could not find any game(s) for "
            f"{game_datetime.year:04d}-{game_datetime.month:02d}"
            f"-{game_datetime.day:02d}. "
            f"If you believe this is an error, "
            f"please raise an issue at "
            f"\n https://github.com/armstjc/ncaa_stats_py/issues \n"
        )
        
    return schedule_df

def get_full_soccer_schedule(
    season: int,
    level: str | int = "I",
    get_womens_soccer_data: bool = False
) -> pd.DataFrame:
    # Placeholder: implement full season aggregation
    pass

def get_soccer_team_roster(
    team_id: int, 
    season: int, 
    get_womens_soccer_data: bool = False
) -> pd.DataFrame:
    """
    Retrieves a soccer team's roster from a given team ID.

    Parameters
    ----------
    `team_id` (int, mandatory):
        Required argument.
        Specifies the team you want a roster from.
        This is separate from a school ID, which identifies the institution.
        A team ID should be unique to a school, and a season.

    `season` (int, mandatory):
        Required argument.
        The season year (e.g., 2025 for Fall 2025 season).

    `get_womens_soccer_data` (bool, optional):
        Optional argument (default: False).
        If you want women's soccer data instead of men's soccer data,
        set this to `True`.

    Usage
    ----------
    ```python

    from ncaa_stats_py.soccer import get_soccer_team_roster

    ########################################
    #          Men's soccer                #
    ########################################

    # Get the soccer roster for the
    # 2024 UNC MLA team (D1, ID: 571437).
    print(
        "Get the soccer roster for the " +
        "2024 UNC MSO team (D1, ID: 571437)."
    )
    df = get_soccer_team_roster(571437, 2024)
    print(df)

    # Get the soccer roster for the
    # 2023 Duke MSO team (D1, ID: 546974).
    print(
        "Get the soccer roster for the " +
        "2023 Duke MSO team (D1, ID: 546974)."
    )
    df = get_soccer_team_roster(546974, 2023)
    print(df)

    ########################################
    #          Women's soccer              #
    ########################################

    # Get the soccer roster for the
    # 2024 Stanford WSO team (D1, ID: 571908).
    print(
        "Get the soccer roster for the " +
        "2024 Stanford WSO team (D1, ID: 571908)."
    )
    df = get_soccer_team_roster(571908, 2024, get_womens_soccer_data=True)
    print(df)

    # Get the soccer roster for the
    # 2023 UCLA WSO team (D1, ID: 546455).
    print(
        "Get the soccer roster for the " +
        "2023 UCLA WSO team (D1, ID: 546455)."
    )
    df = get_soccer_team_roster(546455, 2023, get_womens_soccer_data=True)
    print(df)

    ```

    Returns
    ----------
    A pandas `DataFrame` object with
    an NCAA soccer team's roster for that season.
    """
    # Set sport_id based on parameter
    sport_id = "WSO" if get_womens_soccer_data else "MSO"
    
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

    # Try to get team info from teams cache
    team_info_found = False
    ncaa_division = 1
    ncaa_division_formatted = "I"
    team_conference_name = ""
    school_name = None
    school_id = None

    try:
        # Load teams for the specified sport and find the team
        teams_df = get_soccer_teams(season, "I", get_womens_soccer_data=get_womens_soccer_data)
        if not teams_df.empty:
            team_row = teams_df[teams_df["team_id"] == team_id]
            if not team_row.empty:
                ncaa_division = team_row["ncaa_division"].iloc[0]
                ncaa_division_formatted = team_row["ncaa_division_formatted"].iloc[0]
                team_conference_name = team_row["team_conference_name"].iloc[0] if pd.notna(team_row["team_conference_name"].iloc[0]) else ""
                school_name = team_row["school_name"].iloc[0]
                school_id = int(team_row["school_id"].iloc[0]) if pd.notna(team_row["school_id"].iloc[0]) else None
                team_info_found = True
                logging.info(f"Found team {team_id} in Division {ncaa_division_formatted} teams cache")
        
        # If not found in Division I, try other divisions
        if not team_info_found:
            for div in ["II", "III"]:
                teams_df = get_soccer_teams(season, div, get_womens_soccer_data=get_womens_soccer_data)
                if not teams_df.empty:
                    team_row = teams_df[teams_df["team_id"] == team_id]
                    if not team_row.empty:
                        ncaa_division = team_row["ncaa_division"].iloc[0]
                        ncaa_division_formatted = team_row["ncaa_division_formatted"].iloc[0]
                        team_conference_name = team_row["team_conference_name"].iloc[0] if pd.notna(team_row["team_conference_name"].iloc[0]) else ""
                        school_name = team_row["school_name"].iloc[0]
                        school_id = int(team_row["school_id"].iloc[0]) if pd.notna(team_row["school_id"].iloc[0]) else None
                        team_info_found = True
                        logging.info(f"Found team {team_id} in Division {ncaa_division_formatted} teams cache")
                        break
                        
    except Exception as e:
        logging.warning(f"Could not find team in teams cache: {e}")

    # Ensure cache directories exist
    import os
    base_cache_dir = f"{home_dir}/.ncaa_stats_py"
    soccer_cache_dir = f"{base_cache_dir}/soccer_{sport_id}"
    rosters_cache_dir = f"{soccer_cache_dir}/rosters"
    
    for d in [base_cache_dir, soccer_cache_dir, rosters_cache_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    cache_file = f"{rosters_cache_dir}/{team_id}_roster.csv"

    # Check if we should load from cache
    if os.path.exists(cache_file):
        roster_df = pd.read_csv(cache_file)
        file_mod_datetime = datetime.fromtimestamp(os.path.getmtime(cache_file))
    else:
        file_mod_datetime = datetime.today()
        load_from_cache = False

    now = datetime.today()
    age = now - file_mod_datetime

    # Cache is valid for 14 days for current season teams
    if age.days >= 14 and season >= now.year:
        load_from_cache = False

    if load_from_cache:
        return roster_df

    # Scrape the roster from the team page
    response = _get_webpage(url=url)
    soup = BeautifulSoup(response.text, features="lxml")
    
    # Get school name from page if we don't have it from cache
    if not school_name:
        try:
            school_name = soup.find("div", {"class": "card"}).find("img").get("alt")
        except Exception:
            try:
                school_name = soup.find("div", {"class": "card"}).find("a").text
                school_name = school_name.rsplit(" ", maxsplit=1)[0]
            except Exception:
                school_name = f"Team {team_id}"

    # Get season name
    try:
        season_name = (
            soup.find("select", {"id": "year_list"})
            .find("option", {"selected": "selected"})
            .text
        )
    except Exception:
        # Fallback season name format
        season_name = f"{season}-{str(season+1)[-2:]}"

    # Find the roster table
    try:
        table = soup.find("table", {"class": "dataTable small_font"})
        if table is None:
            table = soup.find("table", {"class": "dataTable small_font no_padding"})
        if table is None:
            # Try finding any table with roster data
            table = soup.find("table")
            
        if table is None:
            logging.warning(f"Could not find roster table for team {team_id}")
            return pd.DataFrame()

        table_headers = table.find("thead").find_all("th")
        table_headers = [x.text.strip() for x in table_headers]

    except Exception as e:
        logging.error(f"Could not parse roster table headers: {e}")
        return pd.DataFrame()

    # Parse roster rows
    try:
        tbody = table.find("tbody")
        if tbody is None:
            t_rows = table.find_all("tr")[1:]  # Skip header row
        else:
            t_rows = tbody.find_all("tr")

        for t in t_rows:
            t_cells = t.find_all("td")
            if len(t_cells) == 0:
                continue
                
            t_cells_text = [x.text.strip() for x in t_cells]

            # Skip empty rows or header rows
            if len(t_cells_text) == 0 or all(cell == "" for cell in t_cells_text):
                continue

            # Handle the specific NCAA soccer roster format
            # Expected columns: GP, GS, #, Name, Class, Position, Height, Hometown, High School
            if len(t_cells_text) >= len(table_headers):
                temp_df = pd.DataFrame(
                    data=[t_cells_text[:len(table_headers)]],
                    columns=table_headers
                )
            else:
                # Pad with empty strings if row has fewer cells
                padded_data = t_cells_text + [""] * (len(table_headers) - len(t_cells_text))
                temp_df = pd.DataFrame(
                    data=[padded_data],
                    columns=table_headers
                )

            # Get player ID and URL from the link in the Name column
            try:
                # Look for the player link in the name cell (usually 4th column, index 3)
                name_cell = t_cells[3] if len(t_cells) > 3 else None
                if name_cell:
                    player_link = name_cell.find("a")
                    if player_link:
                        player_href = player_link.get("href")
                        temp_df["player_url"] = f"https://stats.ncaa.org{player_href}"
                        
                        # Extract player ID from URL (/players/10001471 -> 10001471)
                        player_id = player_href.replace("/players/", "").replace("/", "")
                        player_id = int(player_id)
                        temp_df["player_id"] = player_id
                        
                        # Also get the clean player name from the link text
                        player_name = player_link.text.strip()
                        if player_name:
                            temp_df["Name"] = player_name
                    else:
                        temp_df["player_url"] = None
                        temp_df["player_id"] = None
                else:
                    temp_df["player_url"] = None
                    temp_df["player_id"] = None
            except (ValueError, AttributeError, IndexError):
                temp_df["player_url"] = None
                temp_df["player_id"] = None

            roster_df_arr.append(temp_df)

    except Exception as e:
        logging.error(f"Could not parse roster table rows: {e}")
        return pd.DataFrame()

    if not roster_df_arr:
        logging.warning(f"No roster data found for team {team_id}")
        return pd.DataFrame()

    # Combine all player rows
    roster_df = pd.concat(roster_df_arr, ignore_index=True)
    roster_df = roster_df.infer_objects()

    # Add team metadata
    roster_df["season"] = season
    roster_df["season_name"] = season_name
    roster_df["ncaa_division"] = ncaa_division
    roster_df["ncaa_division_formatted"] = ncaa_division_formatted
    roster_df["team_conference_name"] = team_conference_name
    roster_df["school_id"] = school_id
    roster_df["school_name"] = school_name
    roster_df["sport_id"] = sport_id

    # Standardize column names to match expected format
    column_mapping = {
        "GP": "player_G",
        "GS": "player_GS", 
        "G": "player_G",
        "#": "player_jersey_num",
        "No.": "player_jersey_num",
        "Jersey": "player_jersey_num",
        "Name": "player_full_name",
        "Player": "player_full_name",
        "Class": "player_class",
        "Cl": "player_class",
        "Year": "player_class",
        "Position": "player_positions",
        "Pos": "player_positions",
        "Height": "player_height_string",
        "Ht": "player_height_string",
        "Weight": "player_weight",
        "Wt": "player_weight",
        "Hometown": "player_hometown",
        "Home Town": "player_hometown",
        "High School": "player_high_school",
        "Previous School": "player_high_school",
        "Prev School": "player_high_school",
    }
    
    roster_df.rename(columns=column_mapping, inplace=True)

    # Split full name into first and last name
    if "player_full_name" in roster_df.columns:
        name_split = roster_df["player_full_name"].str.split(" ", n=1, expand=True)
        if name_split.shape[1] >= 2:
            roster_df["player_first_name"] = name_split[0]
            roster_df["player_last_name"] = name_split[1]
        else:
            roster_df["player_first_name"] = name_split[0] if name_split.shape[1] > 0 else ""
            roster_df["player_last_name"] = ""
    
    # Convert GP and GS to numeric, handling non-numeric values
    for col in ["player_G", "player_GS"]:
        if col in roster_df.columns:
            roster_df[col] = pd.to_numeric(roster_df[col], errors='coerce')

    # Convert jersey number to numeric
    if "player_jersey_num" in roster_df.columns:
        roster_df["player_jersey_num"] = pd.to_numeric(roster_df["player_jersey_num"], errors='coerce')

    # Clean up weight column if present
    if "player_weight" in roster_df.columns:
        # Remove non-numeric characters and convert to int
        roster_df["player_weight"] = roster_df["player_weight"].astype(str).str.extract(r'(\d+)').astype(float)

    # Ensure all expected columns exist with proper defaults
    for col in stat_columns:
        if col not in roster_df.columns:
            if col in ["player_G", "player_GS", "player_weight", "player_jersey_num"]:
                roster_df[col] = None  # Numeric columns
            elif col == "player_id":
                roster_df[col] = None  # Keep as None for missing IDs
            else:
                roster_df[col] = ""  # String columns

    # Validate columns and reorder
    final_columns = []
    for col in stat_columns:
        if col in roster_df.columns:
            final_columns.append(col)
        else:
            logging.warning(f"Expected column {col} not found in roster data")

    # Only keep expected columns
    roster_df = roster_df.reindex(columns=final_columns)
    roster_df = roster_df.infer_objects()

    # Save to cache
    roster_df.to_csv(cache_file, index=False)

    return roster_df

def get_soccer_player_season_stats(
    team_id: int,
    season: int,
    level: str | int,
    get_womens_soccer_data: bool = False
) -> pd.DataFrame:
    pass


def get_soccer_player_game_stats(player_id: int) -> pd.DataFrame:
    pass


def get_soccer_game_player_stats(game_id: int) -> pd.DataFrame:
    pass


def get_soccer_raw_pbp(game_id: int) -> pd.DataFrame:
    pass


def get_soccer_team_stats(
    season: int,
    level: str | int = "I",
    get_womens_soccer_data: bool = False
) -> pd.DataFrame:
    pass


if __name__ == "__main__":
    pass