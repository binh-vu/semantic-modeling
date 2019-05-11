use models::relational_schema::RelationalSchema;
use std::path::Path;
use utils::data_reader::load_json;
use utils::data_reader::load_xml;
use serde_json::Value;

pub struct RelationalTable {
    pub id: String,
    pub schema: RelationalSchema,
    pub rows: Vec<Value>
}

impl RelationalTable {

    pub fn load_from_file(fpath: &Path) -> RelationalTable {
        let mut value = match fpath.extension().unwrap().to_str() {
            Some("json") => load_json(fpath),
            Some("xml") => load_xml(fpath),
            Some(ref x) => panic!("Not support extension {}", x),
            None => panic!("Invalid file: {:?}", fpath)
        };

        // we need to have rows in a form of a list of rows, otherwise, we cannot distinguish between whether a property
        // of an entity is a list or a single value
        if value.is_object() {
            while let Value::Object(mut map) = value {
                if map.len() > 1 {
                    panic!("While loading file: {:?}. We try to un-roll a dict to a list or rows, but its have more than one property", fpath);
                }

                unsafe {
                    let key: *const String = map.keys().next().unwrap();
                    value = map.remove(&*key).unwrap();
                }
            }
        }

        let rows = match value {
            Value::Array(vec) => vec,
            _ => panic!("Value should be an array of objects")
        };

        return RelationalTable { id: fpath.file_stem().unwrap().to_str().unwrap().to_owned(), schema: RelationalSchema::build(&rows), rows };
    }
}

//impl fmt::Debug for RelationalTable {
//    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
//        unimplemented!()
//    }
//}