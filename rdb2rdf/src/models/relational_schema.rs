use std::collections::HashMap;
use serde_json::Value;
use std::mem::discriminant;
use std::fmt;
use std::fmt::Formatter;

#[derive(Serialize, Deserialize)]
enum FieldType {
    SingleValue,
    ListValue,
    EmptyList,
    Schema(RelationalSchema),
}

/// Represent an attribute path
#[derive(Serialize, Deserialize)]
pub struct AttrPath {
    attrs: Vec<String>
}

impl AttrPath {
    pub fn new(attrs: Vec<String>) -> AttrPath {
        AttrPath {attrs}
    }
}

pub const PATH_DELIMITER: char = '|';

/// A schema is a json object that define how data is organize in form:
///    <schema> := {<prop> => <value>|<schema>, _type: list|object}
///    <value> := <single_value>|<list_value>
///    <prop> := <string>
///    <single_value> := <token>|<number>
///    <list_value> := list<token>|list<number>
#[derive(Serialize, Deserialize)]
pub struct RelationalSchema {
    attributes: HashMap<String, FieldType>,
    is_list: bool,
}

impl RelationalSchema {
    pub fn build(rows: &Vec<Value>) -> RelationalSchema {
        if rows.len() == 0 {
            panic!("Cannot figure out schema if we don't have any data");
        }

        let mut schema = RelationalSchema::get_schema(&rows[0]);
        for row in rows.iter().skip(1) {
            schema.merge_(RelationalSchema::get_schema(row));
        }

        return schema;
    }

    /// Normalize the data so it will follow its schema
    ///    a property has value None indicate missing values, however, there are some exceptions:
    ///       + a list of objects will be empty list instead of none
    ///       + a dict will be a dict will all property is None (not empty dict)
    ///
    /// the information isn't same, however, in the scope of this project, we ONLY want to figure out the
    /// configuration (semantic mapping); and therefore, the nuance of missing values won't affect to our
    /// decision (only the people who write data transformer need to care); and simply the definition make our
    /// job much easier.
    pub fn normalize(&self, opt_obj: Option<&Value>) -> Value {
        let mut new_obj_val: Value = json!({});
        {
            let new_obj = new_obj_val.as_object_mut().unwrap();
            match opt_obj {
                Some(ref obj) => {
                    for (attr, attr_type) in &self.attributes {
                        match obj.get(attr) {
                            None => {
                                match *attr_type {
                                    FieldType::Schema(ref attr_schema) => {
                                        if attr_schema.is_list {
                                            new_obj.insert(attr.clone(), Value::Array(vec![]));
                                        } else {
                                            new_obj.insert(attr.clone(), attr_schema.normalize(None));
                                        }
                                    },
                                    FieldType::ListValue => { new_obj.insert(attr.clone(), Value::Array(vec![])); },
                                    _ => { new_obj.insert(attr.clone(), Value::Null); }
                                }
                            },
                            Some(attr_val) => {
                                match *attr_type {
                                    FieldType::Schema(ref attr_schema) => {
                                        let new_attr_val = if attr_schema.is_list {
                                            match *attr_val {
                                                Value::Array(ref vs) => Value::Array(vs.iter().map(
                                                    |ref vi| attr_schema.normalize(Some(vi))
                                                ).collect()),
                                                _ => Value::Array(vec![attr_schema.normalize(Some(attr_val))])
                                            }
                                        } else {
                                            attr_schema.normalize(Some(attr_val))
                                        };

                                        new_obj.insert(attr.clone(), new_attr_val);
                                    },
                                    FieldType::ListValue => {
                                        let new_attr_val = if let Value::Array(ref vs) = *attr_val {
                                            attr_val.clone()
                                        } else {
                                            Value::Array(vec![attr_val.clone()])
                                        };

                                        new_obj.insert(attr.clone(), new_attr_val);
                                    },
                                    _ => {
                                        new_obj.insert(attr.clone(), attr_val.clone());
                                    }
                                }
                            }
                        }
                    }
                }
                None => {
                    for (attr, attr_type) in &self.attributes {
                        match *attr_type {
                            FieldType::Schema(ref x) => {
                                let attr_val = if x.is_list { Value::Array(vec![]) } else { Value::Null };
                                new_obj.insert(attr.clone(), attr_val);
                            },
                            _ => {
                                new_obj.insert(attr.clone(), Value::Null);
                            }
                        }
                    }
                }
            }
        }

        return new_obj_val;
    }

    pub fn get_attr_paths(&self) -> Vec<String> {
        let paths = Vec::new();
//        for (attr, attr_type) in &self.attributes {
//            match attr_type {
//            }
//        }

        return paths;
    }

    /// merge another schema to this schema
    fn merge_(&mut self, another: RelationalSchema) {
        self.is_list = self.is_list || another.is_list;

        for (prop, val) in another.attributes {
            if !self.attributes.contains_key(&prop) {
                self.attributes.insert(prop, val);
            } else {
                if self.attributes[&prop] != val {
                    let prop_type = self.attributes.get_mut(&prop).unwrap();
                    if prop_type == &FieldType::SingleValue && &val == &FieldType::ListValue {
                        *prop_type = FieldType::ListValue;
                    } else if prop_type == &FieldType::EmptyList && &val == &FieldType::SingleValue {
                        *prop_type = FieldType::SingleValue;
                    } else if prop_type == &FieldType::EmptyList && &val == &FieldType::ListValue {
                        *prop_type = FieldType::ListValue;
                    } else if let FieldType::Schema(ref mut x) = *prop_type {
                        match val {
                            FieldType::Schema(y) => x.merge_(y),
                            FieldType::EmptyList => {
                                x.is_list = true;
                            }
                            _ => {
                                panic!("Type for prop: {} is incompatible.", prop);
                            }
                        }
                    } else {
                        panic!("Type for prop: {} is incompatible.", prop);
                    }
                }
            }
        }
    }

    fn get_schema(row: &Value) -> RelationalSchema {
        let mut attributes: HashMap<String, FieldType> = HashMap::new();

        if let Value::Object(ref obj) = *row {
            for (prop, val) in obj.iter() {
                match *val {
                    Value::Object(ref x) => {
                        attributes.insert(prop.clone(), FieldType::Schema(RelationalSchema::get_schema(row)));
                    }
                    Value::Array(ref x) => {
                        if x.len() == 0 {
                            attributes.insert(prop.clone(), FieldType::EmptyList);
                        } else {
                            let x0 = x.first().unwrap();
                            assert!(!x0.is_array(), "#properties cannot be dynamic");
                            if let Value::Object(ref x0obj) = *x0 {
                                let mut nested_schema = RelationalSchema::get_schema(x0);
                                for r in x.iter().skip(1) {
                                    nested_schema.merge_(RelationalSchema::get_schema(r));
                                }
                                nested_schema.is_list = true;
                                attributes.insert(prop.clone(), FieldType::Schema(nested_schema));
                            } else {
                                attributes.insert(prop.clone(), FieldType::ListValue);
                            }
                        }
                    }
                    _ => {
                        attributes.insert(prop.clone(), FieldType::SingleValue);
                    }
                }
            }
        }

        return RelationalSchema { attributes, is_list: false };
    }

    fn fmt_debug(schema: &RelationalSchema, indent: usize) -> String {
        let mut vec = Vec::with_capacity(schema.attributes.len());

        for (attr, val) in &schema.attributes {
            let val_fmt = match *val {
                FieldType::Schema(ref x) => RelationalSchema::fmt_debug(x, indent + 1),
                FieldType::ListValue => "ListValue".to_owned(),
                FieldType::EmptyList => "EmptyList".to_owned(),
                FieldType::SingleValue => "SingleValue".to_owned()
            };

            vec.push(format!("{}: {}", attr, val_fmt));
        }
        
        let indentation = " ".repeat((indent + 1) * 4);
        let content = format!("{{\n{}{}\n}}", indentation, vec.join(&format!(",\n{}", &indentation)));
        if schema.is_list {
            return format!("[{}]", content);
        }
        return content;
    }
}

impl PartialEq for FieldType {
    fn eq(&self, other: &FieldType) -> bool {
        return discriminant(self) == discriminant(other);
    }
}

impl fmt::Debug for RelationalSchema {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        return f.write_str(&RelationalSchema::fmt_debug(self, 0));
    }
}