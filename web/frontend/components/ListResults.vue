<template>
    <v-card outlined class="rounded-lg pa-4">
        <v-row>
            <v-col
                v-for="file_path in images"
                :key="file_path"
                cols="12"
                sm="6"
                md="4"
                lg="2"
                class="d-flex child-flex justify-center"
            >
                <v-img
                    :src="file_path"
                    aspect-ratio="1"
                    max-width="256"
                    class="grey lighten-2"
                >
                    <template #placeholder>
                        <v-row
                            class="fill-height ma-0"
                            align="center"
                            justify="center"
                        >
                            <v-progress-circular color="grey lighten-5" />
                        </v-row>
                    </template>
                </v-img>
            </v-col>
        </v-row>
    </v-card>
</template>

<script>
export default {
    name: 'ListResults',
    props: {
        images: {
            type: Array,
            default: () => []
        },
        isLoading: {
            type: Boolean,
            default: false
        }
    }
}
</script>

<style scoped>
</style>
