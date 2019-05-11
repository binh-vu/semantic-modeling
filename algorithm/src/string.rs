/// Break a string to multi-line string, where number of each line characters is less than a maximum characters
///
/// Treat this as optimization problem, where we trying to minimize the number of line break
/// but also maximize the readability in each line, i.e: maximize the number of characters in each lines
/// Using greedy search.
pub fn auto_wrap(word: &str, max_char_per_line: u32, delimiters: Option<&[char]>, camelcase_split: bool) -> String {
    let default_delimiter = [' ', ':', '_', '/'];
    let delimiters = if delimiters.is_none() {
        &default_delimiter
    } else {
        delimiters.unwrap()
    };

    let mut sublines = vec!["".to_owned()];
    let chars = word.chars().collect::<Vec<_>>();

    for (i, &c) in chars.iter().enumerate() {
        if delimiters.iter().find(|&&x| x == c).is_some() {
            sublines.last_mut().unwrap().push(c);

            if camelcase_split && !c.is_uppercase() && i + 1 < word.len() && chars[i + 1].is_uppercase() {
                // camelcase split
                sublines.push("".to_owned());
            }
        } else {
            sublines.last_mut().unwrap().push(c);
            sublines.push("".to_owned());
        }
    }

    let mut new_sublines = vec!["".to_owned()];
    for line in sublines {
        if new_sublines.last().unwrap().len() + line.len() <= max_char_per_line as usize {
            *new_sublines.last_mut().unwrap() += &line;
        } else {
            new_sublines.push(line);
        }
    }

    new_sublines.join("\n")
}

pub fn left(s: &str, w: usize) -> String {
    let n_padding = w - s.len();
    s.to_owned() + &(0..n_padding).map(|_| " ").collect::<String>()
}

pub fn center(s: &str, w: usize) -> String {
    let n_padding = (w - s.len()) / 2;
    (0..n_padding).map(|_| " ").collect::<String>() + s + &(0..n_padding).map(|_| " ").collect::<String>()
}

pub fn right(s: &str, w: usize) -> String {
    let n_padding = w - s.len();
    (0..n_padding).map(|_| " ").collect::<String>() + s
}

pub fn align_table(rows: &[Vec<String>], align: &[&str]) -> Vec<Vec<String>> {
    let mut col_widths = vec![0; rows[0].len()];
    // compute col widths
    for row in rows {
        for (i, col) in row.iter().enumerate() {
            col_widths[i] = col_widths[i].max(col.len());
        }
    }

    rows.into_iter()
        .map(|row| {
            row.iter().enumerate()
                .map(|(i, col)| {
                    match align[i] {
                        "left" => left(col, col_widths[i]),
                        "right" => right(col, col_widths[i]),
                        "center" => center(col, col_widths[i]),
                        _ => panic!("Invalid align option: {}", align[i]),
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}