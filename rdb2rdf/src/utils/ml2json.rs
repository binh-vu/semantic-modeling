use serde_json::Value;
use std::io::Read;
use xml::EventReader;
use xml::reader::XmlEvent;
use yaml_rust::Yaml;
use serde_json::value::Value::Number;

fn get_empty_object() -> Value {
    return json!({});
}

const TEXT_LBL: &str = "#text";

pub fn xml2json<R: Read>(parser: EventReader<R>) -> Value {
    let mut root: Value = json!({});
    let mut level = 0;
    {
        let mut paths = vec![&mut root];
        for e in parser {
            match e {
                Ok(XmlEvent::StartElement { name, attributes, .. }) => {
                    // add new attributes
                    let mut new_node = get_empty_object();
                    {
                        let new_node_val = new_node.as_object_mut().unwrap();
                        for attr in attributes.iter() {
                            new_node_val.insert(format!("@{}", attr.name.local_name), Value::String(attr.value.clone()));
                        }
                    }

                    // add new attribute to current path
                    let x: *mut Value;
                    {
                        let mut obj = paths[level].as_object_mut().unwrap();

                        if obj.contains_key(&name.local_name) {
                            // add new value to array;
                            let prev_vals = obj.get_mut(&name.local_name).unwrap();
                            if prev_vals.is_array() {
                                let n_elements = prev_vals.as_array().unwrap().len();
                                prev_vals.as_array_mut().unwrap().push(new_node);
                                x = &mut prev_vals[n_elements];
                            } else {
                                *prev_vals = Value::Array(vec![new_node]);
                                x = &mut prev_vals[0];
                            }
                        } else {
                            obj.insert(name.local_name.clone(), new_node);
                            x = obj.get_mut(&name.local_name).unwrap();
                        }
                    }
                    unsafe { paths.push(&mut *x); }
                    level += 1;
                }
                Ok(XmlEvent::EndElement { name, .. }) => {
                    let current_node = paths.pop().unwrap().as_object_mut().unwrap();
                    level -= 1;

                    if current_node.len() == 1 && current_node.contains_key(TEXT_LBL) {
                        let text = current_node.remove(TEXT_LBL).unwrap();
                        paths[level].as_object_mut().unwrap().insert(name.local_name.clone(), text);
                    }
                }
                Ok(XmlEvent::Characters(content)) => {
                    paths[level].as_object_mut().unwrap().insert(String::from(TEXT_LBL), Value::String(content));
                }
                _ => {}
            }
        }
    }

    return root;
}

fn _yml2json(doc: Yaml) -> Value {
    match doc {
        Yaml::Real(x) => json!(x.parse::<f64>().unwrap()),
        Yaml::Null => Value::Null,
        Yaml::String(x) => json!(x),
        Yaml::Array(arr) => Value::Array(arr.into_iter().map(|a| _yml2json(a)).collect()),
        Yaml::Boolean(b) => Value::Bool(b),
        Yaml::Hash(subdoc) => {
            let mut o: Value = json!({});
            {
                let mut oval = o.as_object_mut().unwrap();
                for (k, v) in subdoc.into_iter() {
                    match k {
                        Yaml::String(s) => oval.insert(s, _yml2json(v)),
                        _ => panic!("Invalid YAML")
                    };
                }
            }
            o
        },
        Yaml::Integer(x) => json!(x),
        _ => panic!("Not support: {:?}", doc)
    }
}

pub fn yml2json(docs: Vec<Yaml>) -> Value {
    return Value::Array(docs.into_iter().map(|d| _yml2json(d)).collect());
}