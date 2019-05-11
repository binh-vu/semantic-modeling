from slugify import slugify
from dateutil.parser import *


class Misc:

    @staticmethod
    def slugify(text: str) -> str:
        return slugify(text)

    @staticmethod
    def is_year(text: str) -> bool:
        return len(text) == 4 and text.isdigit()

    @staticmethod
    def parse_noisy_birth_and_death_year(birth_and_death_years: str) -> (str, str):
        if birth_and_death_years == "":
            return "", ""

        if birth_and_death_years.find("-") != -1:
            birth, death = birth_and_death_years.split("-", maxsplit=1)
            return birth.strip(), death.strip()

        return "", ""

    @staticmethod
    def get_date_earliest(date: str) -> str:
        if date:
            if Misc.is_year(date):
                return date + "-01-01"
            else:
                try:
                    return parse(date).strftime("%Y-%m-%d")
                except:
                    return ""
        return ""

    @staticmethod
    def get_date_latest(date: str) -> str:
        if date:
            if Misc.is_year(date):
                return date + "-12-31"
            else:
                try:
                    return parse(date).strftime("%Y-%m-%d")
                except:
                    return ""
        return ""


class ArtistHelper:

    @staticmethod
    def parse_born_die_date_npg(born_die_date: str) -> (str, str):
        if not born_die_date:
            return "", ""

        if born_die_date.find("-") != -1:
            born, die = born_die_date.split("-", maxsplit=1)
            return born, die
        else:
            if born_die_date.strip().startswith("born"):
                return born_die_date.replace("born ", ""), ""
        return "", ""

    @staticmethod
    def get_artist_uri(url: str, possible_id: str) -> str:
        if possible_id:
            if url.endswith("/"):
                return url + possible_id
            return url + "/" + possible_id
        return ""

    @staticmethod
    def get_birth_uri(artist_uri: str, birth_date: str) -> str:
        if artist_uri and birth_date:
            return artist_uri + "/birth"
        return ""

    @staticmethod
    def get_birth_date_uri(artist_uri: str, birth_date: str) -> str:
        if artist_uri and birth_date:
            return artist_uri + "/birth_date"
        return ""

    @staticmethod
    def get_death_uri(artist_uri: str, death_date: str) -> str:
        if artist_uri and death_date:
            return artist_uri + "/death"
        return ""

    @staticmethod
    def get_death_date_uri(artist_uri: str, death_date: str) -> str:
        if artist_uri and death_date:
            return artist_uri + "/death_date"
        return ""

    @staticmethod
    def get_nationality_uri(artist_uri: str, nationality: str) -> str:
        if artist_uri and nationality:
            return artist_uri + "/nationality"
        return ""


class ArtworkHelper:

    @staticmethod
    def get_object_uri(url: str, object_unique_identifier: str) -> str:
        if object_unique_identifier.strip():
            if url.endswith("/"):
                return url + object_unique_identifier
            return url + "/" + object_unique_identifier
        return ""

    @staticmethod
    def get_accession_uri(object_uri: str) -> str:
        if object_uri:
            return object_uri + "/accession_number"
        return ""

    @staticmethod
    def get_object_id_uri(object_uri: str) -> str:
        if object_uri:
            return object_uri + "/object_id"
        return ""

    @staticmethod
    def get_production_uri(object_uri: str) -> str:
        if object_uri:
            return object_uri + "/production"
        return ""

    @staticmethod
    def get_primary_title_uri(object_uri: str, title: str) -> str:
        if object_uri and title:
            return object_uri + "/primary_title"
        return ""

    @staticmethod
    def get_product_date_uri(object_uri: str, date: str) -> str:
        if object_uri and date:
            return object_uri + "/production/date"
        return ""

    @staticmethod
    def get_medium_uri(object_uri: str, medium: str) -> str:
        if object_uri and medium:
            return object_uri + "/production/medium"
        return ""

    @staticmethod
    def get_dimension_uri(object_uri: str, dimension: str) -> str:
        if object_uri and dimension:
            return object_uri + "/production/dimension"
        return ""

    @staticmethod
    def get_type_assignment_uri(object_uri: str, work_type: str) -> str:
        if object_uri and work_type:
            return object_uri + "/work_type"
        return ""

    @staticmethod
    def get_acquisition_uri(object_uri: str, acquisition_time: str) -> str:
        if object_uri and acquisition_time:
            return object_uri + '/acquisition'
        return ""

    @staticmethod
    def get_creator_uri(object_uri: str, artist_name: str) -> str:
        if object_uri and artist_name:
            return object_uri + "/creator"
        return ""

    @staticmethod
    def get_sitter_uri(object_uri: str, artist_name: str) -> str:
        if object_uri and artist_name:
            return object_uri + "/sitter"
        return ""


if __name__ == '__main__':
    pass