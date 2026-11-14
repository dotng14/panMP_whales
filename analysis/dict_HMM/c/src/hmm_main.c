#include "dict_hmm.h"

#include <stdio.h>
#include <stdlib.h>

/* Read "motif start_sec end_sec" lines and write the two viewer tables.
   argv: placements.txt segment.csv motif_segments.csv */

static int theme_of(const DhThemes *themes, double start_sec) {
    double tmin = start_sec / 60.0;
    int which = 0;
    for (int i = 0; i < themes->n_themes; i++) {
        double a = themes->themes[i].start_sec / 60.0;
        double b = themes->themes[i].end_sec / 60.0;
        if (a <= tmin && tmin < b) which = i;
    }
    return themes->themes[which].index;
}

static int song_of(const DhThemes *themes, double start_sec) {
    for (int i = 0; i < themes->n_songs; i++) {
        if (themes->songs[i].start_sec <= start_sec && start_sec < themes->songs[i].end_sec)
            return themes->songs[i].index;
    }
    return themes->n_songs ? themes->songs[themes->n_songs - 1].index : 1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s placements.txt segment.csv motif_segments.csv\n", argv[0]);
        return 2;
    }
    FILE *in = fopen(argv[1], "r");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int cap = 256, n = 0;
    int *ids = malloc((size_t)cap * sizeof(int));
    double *starts = malloc((size_t)cap * sizeof(double));
    double *ends = malloc((size_t)cap * sizeof(double));
    int id;
    double a, b;
    while (fscanf(in, "%d %lf %lf", &id, &a, &b) == 3) {
        if (n == cap) {
            cap *= 2;
            ids = realloc(ids, (size_t)cap * sizeof(int));
            starts = realloc(starts, (size_t)cap * sizeof(double));
            ends = realloc(ends, (size_t)cap * sizeof(double));
        }
        ids[n] = id;
        starts[n] = a;
        ends[n] = b;
        n++;
    }
    fclose(in);
    DhThemes themes;
    if (dict_hmm_themes(ids, starts, ends, n, &themes) != 0) return 1;

    FILE *seg = fopen(argv[2], "w");
    FILE *mot = fopen(argv[3], "w");
    if (!seg || !mot) {
        perror("csv");
        return 1;
    }
    fprintf(seg, "type,segment index,start_unix,end_unix\n");
    fprintf(mot, "motif,song,theme,phrase_id,phrase_count,start_unix,end_unix\n");
    for (int i = 0; i < n; i++) {
        int theme = theme_of(&themes, starts[i]);
        int song = song_of(&themes, starts[i]);
        fprintf(seg, "motif,%d,%.6f,%.6f\n", ids[i],
                DICT_HMM_WAV2_UNIX + starts[i], DICT_HMM_WAV2_UNIX + ends[i]);
        fprintf(mot, "%d,%d,%d,%d,1,%.6f,%.6f\n", ids[i], song, theme, theme,
                DICT_HMM_WAV2_UNIX + starts[i], DICT_HMM_WAV2_UNIX + ends[i]);
    }
    for (int i = 0; i < themes.n_themes; i++) {
        fprintf(seg, "theme,%d,%.6f,%.6f\n", themes.themes[i].index,
                DICT_HMM_WAV2_UNIX + themes.themes[i].start_sec,
                DICT_HMM_WAV2_UNIX + themes.themes[i].end_sec);
    }
    for (int i = 0; i < themes.n_songs; i++) {
        fprintf(seg, "song,%d,%.6f,%.6f\n", themes.songs[i].index,
                DICT_HMM_WAV2_UNIX + themes.songs[i].start_sec,
                DICT_HMM_WAV2_UNIX + themes.songs[i].end_sec);
    }
    fclose(seg);
    fclose(mot);
    fprintf(stderr, "placements %d themes %d songs %d\n", n, themes.n_themes, themes.n_songs);
    for (int i = 0; i < themes.n_songs; i++) {
        fprintf(stderr, "song %d %.1f-%.1f\n", themes.songs[i].index,
                themes.songs[i].start_sec / 60.0, themes.songs[i].end_sec / 60.0);
    }
    dict_hmm_free_themes(&themes);
    free(ids);
    free(starts);
    free(ends);
    return 0;
}
