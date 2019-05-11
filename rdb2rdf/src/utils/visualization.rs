use serde_json::Value;
use prettytable::Table;
use models::relational_table::RelationalTable;

fn prettify(rows: &Vec<Value>) {
    let table = Table::new();
}

pub fn tbl2ascii(tbl: &RelationalTable) {
    prettify(&tbl.rows);
}