find out -path 'out/*/*_output.csv' -type f \
  -exec awk '
    { last = $0 }
    ENDFILE {
      if (last !~ /^vocabulary/) {
        print FILENAME
      }
    }
  ' {} +
